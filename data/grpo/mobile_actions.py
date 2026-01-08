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


def mobileactions(tokenizer, prompt_token_len):
    apply_chat_template = lambda seq: tokenizer.apply_chat_template(
                seq,
                add_generation_prompt=False,
                tokenize=False,
                continue_final_message=True,
            )
    
    def mapper(data):
        tools = data["tools"] #json.loads(data["tools"])
        tools = [t.get('function', t) for t in tools]
        if not isinstance(tools, list):
            tools = [tools]

        messages = data['messages'] #json.loads(data['messages'])
        date_time = ' '.join(messages[0]['content'].split('\n')[:2])
        user_question = messages[1]['content']

        tool_calls = messages[2]['tool_calls']
        tool_calls = [t.get('function', t) for t in tool_calls]
        called_tools_name = [t['name'] for t in tool_calls]

        random.shuffle(tools)
        del_idx = []
        for i, t in enumerate(tools):
            if t['name'] in called_tools_name:
                continue
            del_idx.append(i)
            if len(del_idx) == 3: break
        
        del_idx = sorted(del_idx, reverse=True)
        for di in del_idx:
            del tools[di]

        seq = [
            {
                "role": "system",
                "content": TOOL_TEMPLATE.format(tools=tool_shuffle(tools)) + '\n' + str(date_time)
            },
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": "<tool_call>"}
        ]
        
        return {
            "prompt": apply_chat_template(seq),
            "messages": seq,
            "def_tools": tools,
            "ground_tool_call": json.dumps(tool_calls, default=str),
            "num_input_tools": len(tools),
            "scorer": partial(scorer, tools_ground=tool_calls, def_tools=tools)
        }

    train_ds = load_dataset("google/mobile-actions")["train"]
    train_ds = list(map(mapper, train_ds))
    train_ds = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, train_ds))
    return train_ds


from .verifiers import validate_format, tool_scorer, thinking_validate


def scorer(llm_gen, tools_ground, def_tools):
    # Adding think tag (prefilled in dataset)
    # Tool score
    llm_gen = "<tool_call>" + llm_gen
    tool_score, tools_gen = tool_scorer(llm_gen, tools_ground, def_tools)
    return tool_score


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

    ds = mobileactions(tokenizer, prompt_token_len=768)
    fc_names = []
    print("Dataset length:", len(ds))

    print(ds[0]['prompt'])
    print(ds[0])