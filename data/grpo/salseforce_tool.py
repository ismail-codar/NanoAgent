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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
k_shot_example = [
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

def salesfores_toolcall(tokenizer, prompt_token_len, n_tool_calls=2, n_tool_inputs=4, dedupe_ratio=0.995, think=False, k_shot=False):
    
    apply_chat_template = lambda seq: tokenizer.apply_chat_template(
                seq,
                add_generation_prompt=False if think else True,
                tokenize=False,
                continue_final_message=True if think else False,
            )
    
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
            "prompt": apply_chat_template(seq),
            "messages": seq,
            "def_tools": tools,
            "ground_tool_call": tool_calls,
            "num_input_tools": len(tools),
            "scorer": partial(scorer, tools_ground=tool_calls, def_tools=tools, think=think)
        }

    train_ds = load_dataset("Salesforce/xlam-function-calling-60k")["train"]
    train_ds = list(filter(lambda x: 0 < len(x["ground_tool_call"]) <= n_tool_calls and 1 < x["num_input_tools"] <= n_tool_inputs, map(mapper, train_ds)))
    semhash = SemHash.from_records(train_ds, columns=['prompt'], use_ann=False)
    train_ds = semhash.self_deduplicate(threshold=dedupe_ratio)
    train_ds = train_ds.selected

    train_ds = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, train_ds))

    if k_shot:
        for i in range(len(train_ds)):
            # Randomly sample one example
            rand_ds = random.choice(train_ds)
            train_ds[i]['messages'] = rand_ds[i]['messages'] + train_ds[i]['messages']
            train_ds[i]['prompt'] = apply_chat_template(train_ds[i]['messages'])
        train_ds = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, train_ds))

    print("Input tool distribution:", np.bincount([x["num_input_tools"] for x in train_ds]))
    print(
        "Tool call distribution:",
        np.bincount([len(x["ground_tool_call"]) for x in train_ds]),
    )

    return train_ds


from .verifiers import validate_format, tool_scorer, thinking_validate


def scorer(llm_gen, tools_ground, def_tools, think=True):
    # Adding think tag (prefilled in dataset)
    if think:
        llm_gen = "<think>" + llm_gen
        # Validate format
        valid_format = validate_format(llm_gen)
        # print("Invalid format:", llm_gen, flush=True)
        if not valid_format:
            return -1
    
    # Tool score
    tool_score, tools_gen = tool_scorer(llm_gen, tools_ground, def_tools)
    if not think:
        return tool_score
    
    if tool_score <= 0:
        return tool_score

    # think_score = thinking_scorer(llm_gen, tools_gen, def_tools)
    think_score = int(thinking_validate(llm_gen))
    # if think_score <= 0:
        # print("Invalid thinking", flush=True)
        # return tool_score

    return tool_score + think_score


if __name__ == '__main__':
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collections import Counter

    tokenizer = AutoTokenizer.from_pretrained("weights/NanoAgent-135M")
    def total_tokens(data):
        return len(
            tokenizer.encode(
                data,
            )  # add_generation_prompt=True)
        )

    ds = salesfores_toolcall(tokenizer, dedupe_ratio=0.95)
    # ds = list(filter(lambda x: total_tokens(x['prompt']) <= 1024+128, ds))
    fc_names = []
    for d in ds:
        fc_names.append(d['ground_tool_call'][0]['name'])
    print("Dataset length:", len(ds))
    # print(Counter(fc_names))