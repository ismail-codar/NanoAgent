import json
import re
from difflib import SequenceMatcher
from functools import partial
import random

from utils.tokenizer import TOOL_TEMPLATE
from utils.webtool import tool_call_extract
from data.utils import THINK_STRINGS

import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_MODEL = "qwen3:0.6b"
# OLLAMA_MODEL = "gemma3:270m"

def response_judge(question, response, n_tokens):
    judge_prompt = \
f'''You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial, or strongly incorrect
2: The system_answer is mostly not helpful: misses some key instructions of the question, repeats steps needlessly, answer is mostly incorrect
3: The system_answer is mostly helpful: provides support follows most instructions, but still could be improved, answer can be somewhat incorrect
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the instructions in the question

Provide your feedback as follows:

Feedback:::
Evaluation: (your extremely concise rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

# Question:\n{question}

# Answer:\n{response}

---

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
'''

    response = get_llm_response(
        messages=[{'role': 'user', 'content': judge_prompt}],
        think=False,
        n_tokens=n_tokens
    )[1]

    last_line = list(filter(lambda x: len(x.strip()) > 0, response.split('\n')))[-1].strip()
    digits = re.findall(r'\d+', last_line)
    if digits:
        score = int(digits[-1]) / 4
    else:
        score = 0

    # print(judge_prompt)
    # print("RESPONSE")
    # print(response)
    # print(score)
    # print('---')

    return response, score


def get_llm_response(messages, think=False, n_tokens=2):
    if think:
        assert n_tokens >= 8, 'Number of generating tokens must be higher if reasoning/think is set to True'
    ret = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=False,
        think=think,
        options={
            # 'ctx': 512,
            'temperature': 0.,
            'num_predict': n_tokens
        },   
    )

    return ret.message.thinking, ret.message.content


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
    except Exception as E:
        print("Exception:", E, '| Input:', llm_gen)
        return 0, None


def _tool_scorer(llm_gen, tools_ground, def_tools, threshold, verbose=False):
    def uniform(s:str):
        return sorted(list(s.lower()))

    if verbose:
        print("Gen tools:", type(llm_gen), json.dumps(llm_gen))
        print("Ground tools:", type(tools_ground), json.dumps(tools_ground))

    assert isinstance(llm_gen, str)
    tools_gen = tool_call_extract(llm_gen)
    if verbose:
        print("Parsed toolcall:", type(tools_gen), json.dumps(tools_gen))
    tool_ground_names = [t['name'] for t in tools_ground]
    tools_ground_args = {t['name']:t['arguments'] for t in tools_ground}
    req_ground_attribs = {t['name']:t.get('parameters', {}).get('required', []) for t in def_tools}
    total_score = 0
    args_score = []

    if tools_gen is None:
        return 0, None
    if len(tools_gen) != len(tools_ground):
        return 0, None

    # Scoring:
    # -2: Major mistakes -> Wrong format | imaginary tools | invalid tool call signature
    # -1: Minor mistakes -> wrong arguments | 

    for tool in tools_gen:
        # Invalid tool calling format
        if not isinstance(tool, dict):
            return 0, None
        # Name/args missing
        if 'name' not in tool or 'arguments' not in tool:
            return 0, None
        # Not a dict
        if not isinstance(tool['arguments'], dict):
            return 0, None
        # Invalid tool call
        if tool['name'] not in tool_ground_names:
            return 0, None
        # Invalid arguments
        for param_name, gen_val in tool['arguments'].items():
            if param_name not in tools_ground_args[tool['name']]:
                return 0, None
            ground_val = tools_ground_args[tool['name']][param_name]

            sim_score = cosine_similarity_tfidf(str(gen_val), str(ground_val))
            args_score.append(sim_score if type(gen_val) is type(ground_val) else 0)
            
        # TODO: Missing required attribs
        for param_name in req_ground_attribs[tool['name']]:
            if param_name not in tool['arguments']:
                # print(param_name, 'missing in', llm_gen)
                total_score += -0.05

    a = str(tools_gen)
    b = str(tools_ground)

    if uniform(a) == uniform(b):
        return 1, tools_gen

    s = SequenceMatcher(None, a, b)
    seq_match = (s.find_longest_match().size / len(b))
    total_score += (sum(args_score) / len(args_score))
    total_score = seq_match * 0.25 + max(total_score, 0) * 0.75
    return max(min(total_score, 1), 0), tools_gen


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
        model=OLLAMA_MODEL,
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


def cosine_similarity_tfidf(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
    return (cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]).item()


def thinking_scorer(llm_gen, tools_gen, def_tools):
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think = pattern.findall(llm_gen)
    if not think or tools_gen is None: return 0
    think = think[0]
    if '?' in think: return 0

    # Length normalize
    matcher = SequenceMatcher(None, think, str(tools_gen))
    
    if len(matcher.get_matching_blocks()) >= 2 * len(tools_gen):
        return 1
    return 0


if __name__ == '__main__':
    pass
#     The best way to invest in crypto currency. Use the phrase "the best of both worlds".<|im_end|>
# <|im_start|>assistant

# 1.00: No, "the best of both worlds" is not the phrase but rather a sentiment that suggests a combination of benefits from both forms of investment, making it ideal for someone looking for a balanced strategy.<|im_end|>
