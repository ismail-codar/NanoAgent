import json
import re
from difflib import SequenceMatcher
from functools import partial
import random

from data.utils import tool_shuffle
from utils.tokenizer import TOOL_TEMPLATE
from utils.webtool import tool_call_extract
from data.utils import THINK_STRINGS

from datasets import load_dataset
from semhash import SemHash
import ollama
import numpy as np

# K-shot Prompt
ws_tool = (
    {
        "name": "web_search",
        "description": "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to perform.",
                }
            },
            "required": ["query"],
        },
    },
)
# Not being used
k_shot = [
    {
        "role": "system",
        "content": TOOL_TEMPLATE.format(tools=json.dumps(ws_tool))
        + " Think before answering.",
    },
    {"role": "user", "content": "What is the capital of Canada?"},
    {
        "role": "assistant",
        "content": """<think>The user is asking to know the capital of Canada. I see I have access to 'web_search' tool. So I can use that to find the capital of Canada. 'web_search' tool requires one parameter 'query'. The value for 'query' should be 'The capital of Canada' in this case.</think>\n\n<tool_call>[{"name": "web_search", "arguments": {"query": "The capital of Canada"}}]</tool_call>""",
    },
]

def salesfores_toolcall(tokenizer, n_tool_calls=2, n_tool_inputs=4, think=True):
    def mapper(data):
        tools = json.loads(data["tools"])
        tool_calls = json.loads(data["answers"])
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tools, list):
            tools = [tools]
        
        seq = [
            {
                "role": "system",
                "content": TOOL_TEMPLATE.format(tools=tool_shuffle(tools))
                + (random.choice(THINK_STRINGS) if think else ""),
            },
            {"role": "user", "content": data["query"]},
        ]
        if think:
            seq.append({"role": "assistant", "content": "<think>"})
        
        return {
            "prompt": tokenizer.apply_chat_template(
                seq,
                add_generation_prompt=False if think else True,
                tokenize=False,
                continue_final_message=True if think else False,
            ),
            "messages": seq,
            "def_tools": tools,
            "ground_tool_call": tool_calls,
            "num_input_tools": len(tools),
            "scorer": partial(scorer, tools_ground=tool_calls, def_tools=tools, think=think)
        }

    train_ds = load_dataset("Salesforce/xlam-function-calling-60k")["train"]
    train_ds = list(filter(lambda x: 0 < len(x["ground_tool_call"]) <= n_tool_calls and 1 < x["num_input_tools"] <= n_tool_inputs, map(mapper, train_ds)))
    semhash = SemHash.from_records(train_ds, columns=['prompt'])
    train_ds = semhash.self_deduplicate(threshold=0.995)
    # print("Dedup ratio:", train_ds.duplicate_ratio)
    train_ds = train_ds.selected

    print("Input tool distribution:", np.bincount([x["num_input_tools"] for x in train_ds]))
    print(
        "Tool call distribution:",
        np.bincount([len(x["ground_tool_call"]) for x in train_ds]),
    )

    return train_ds



def validate_format(text):
    """
    Validate if the text strictly follows the pattern:
    <think> ... </think><tool_call> ... </tool_call>

    Returns True if the string matches the pattern, False otherwise.
    """
    pattern = re.compile(
        r"^<think>.*?</think>\s*<tool_call>.*?</tool_call>$", re.DOTALL
    )
    return bool(pattern.match(text)) \
        and (text.count("</think>") == 1) \
        and (text.count("<tool_call>") == 1) \
        and (text.count("</tool_call>") == 1)


def tool_scorer(llm_gen, tools_ground, def_tools, verbose=False):
    try:
        return _tool_scorer(llm_gen, tools_ground, def_tools, verbose)
    except:
        return -1
    return -1


def _tool_scorer(llm_gen, tools_ground, def_tools, verbose=False):
    def uniform(s:str):
        return sorted(list(s.lower()))

    if verbose:
        print("Gen tools:", type(llm_gen), json.dumps(llm_gen))
        print("Ground tools:", type(tools_ground), json.dumps(tools_ground))

    assert isinstance(llm_gen, str)
    tools_gen = tool_call_extract(llm_gen)
    if verbose:
        print("Parsed toolcall:", type(tools_gen), json.dumps(tools_gen))
    tool_names = [t['name'] for t in tools_ground]
    tools_ground_attribs = {t['name']:t['arguments'] for t in tools_ground}
    req_ground_attribs = {t['name']:t.get('parameters', {}).get('required', []) for t in def_tools}
    total_score = 0

    if tools_gen is None:
        return -1, None
    for tool in tools_gen:
        # Invalid tool calling format
        if not isinstance(tool, dict):
            return -1, None
        if 'name' not in tool or 'arguments' not in tool:
            return -1, None
        if not isinstance(tool['arguments'], dict):
            return -1, None
        if tool['name'] not in tool_names:
            return -1, None
        # Invalid arguments
        for param_name, val in tool['arguments'].items():
            if param_name not in tools_ground_attribs[tool['name']]:
                return -1, None
        # TODO: Missing required attribs
        for param_name in req_ground_attribs[tool['name']]:
            if param_name not in tool['arguments']:
                total_score += -0.25

    a = str(tools_gen)
    b = str(tools_ground)

    if uniform(a) == uniform(b):
        return 2, tools_gen

    s = SequenceMatcher(None, a, b)
    total_score += s.ratio() + (s.find_longest_match().size / len(b))
    return max(total_score, -1), tools_gen


def thinking_validate(llm_gen):
    TOOL_VERIFY_PROMPT = """"You will be given a chain of thought of an LLM that tries to plan and make function/tool call. 
The step by step thinking/planning would be inside <think> </think> tags and the tool/function call would be inside <tool_call> </tool_call> tags. 
You have to validate if the step by step is reflecting the tool calls properly or not.


# Instructions:

- If the 'think' is not reflecting the actions made in inside 'tool_call': return False
- If the 'think' is not contextually idealizing the actions inside 'tool_call': return False
- If the 'think' highlighting different actions than taken inside 'tool_call': return False
- If the 'think' is indicating multiple steps but execute only one tool: return False
- If the 'think' contradicts with actions taken in 'tool_call': return False
- If the 'think' is not reflecting on how to solve the problem using tools: return False

If all of the above instructions pass, return True


# Response Format:

Respond either True of False.
Example:
[True/False]
"""

    # def chat_with_ollama(example):
    messages = [
        # {"role": "system", "content": "You are a helpful AI assistant named SmolThink. First plan/reason/code/validate inside <think></think> tag and provide final answer to user query inside <answer></answer> tag."},
        {"role": "system", "content": TOOL_VERIFY_PROMPT},
        {"role": "user", "content": llm_gen}
    ]

    # First call: Initial chat with streaming enabled
    resp = ollama.chat(
        model="qwen3:0.6b",
        messages=messages,
        stream=False,
        think=False,
        options={
            # 'ctx': 512,
            'temperature': 0.,
            'num_predict': 2
        },   
    )

    return resp.message.content.strip().lower() == 'true'


def thinking_scorer(llm_gen, tools_gen, def_tools):
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think = pattern.findall(llm_gen)
    if not think or tools_gen is None: return -1
    think = think[0]
    if '?' in think: return -1

    # Length normalize
    matcher = SequenceMatcher(None, think, str(tools_gen))
    
    if len(matcher.get_matching_blocks()) >= 2 * len(tools_gen):
        return 1
    return -1


def scorer(llm_gen, tools_ground, def_tools, think=True):
    # Adding think tag (prefilled in dataset)
    if think:
        llm_gen = "<think>" + llm_gen

        # Validate format
        valid_format = validate_format(llm_gen)
        if not valid_format:
            return -1
    
    # Tool score
    tool_score, tools_gen = tool_scorer(llm_gen, tools_ground, def_tools)
    if tool_score <= 0:
        return -1

    if not think:
        return tool_score

    # think_score = thinking_scorer(llm_gen, tools_gen, def_tools)
    think_score = int(thinking_validate(llm_gen))
    if think_score <= 0:
        return -1

    return tool_score + think_score