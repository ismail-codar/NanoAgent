import json
import random
from ast import literal_eval
from copy import deepcopy
from functools import partial
import re
import numpy as np

from datasets import load_dataset

from data.utils import tool_shuffle
from utils.tokenizer import TOOL_TEMPLATE
from utils.tools import parse_tool_calls


def txt360_toolcall(tokenizer, prompt_token_len, dedupe_ratio=None):
    apply_chat_template = lambda seq: tokenizer.apply_chat_template(
        seq,
        add_generation_prompt=True,
        tokenize=False,
        continue_final_message=False,
    )
    
    def mapper(data):
        try:
            data = json.loads(data['messages'])
        except Exception:
            return None

        seq = []
        tools = None
        tool_names = None

        for idx, turn in enumerate(data):
            role = turn.get('role')
            
            if role == 'system':
                try:
                    tools = turn.get('tools', [])
                    if not tools:
                        tools = []
                    if not isinstance(tools, list):
                        tools = [tools]
                    tool_names = [t['name'] for t in tools]
                    seq.append({
                        "role": "system",
                        "content": TOOL_TEMPLATE.format(tools=tool_shuffle(tools))
                    })
                except Exception:
                    return None

            elif role == 'user':
                try:
                    seq.append({'role': 'user', 'content': turn.get('content', '')})
                except Exception:
                    return None

            elif role == 'assistant':
                try:
                    if tools is None:
                        return None
                    
                    tool_calls = turn.get('tool_calls', [])
                    if not tool_calls:
                        return None
                    
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    
                    parsed_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            name = tc.get('name')
                            arguments = tc.get('arguments')
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except:
                                    return None
                            parsed_calls.append({'name': name, 'arguments': arguments})
                        else:
                            return None
                    
                    for tc in parsed_calls:
                        if tc["name"] not in tool_names:
                            return None
                    
                    seq.append({
                        "role": "assistant",
                        "content": f"<tool_call>{json.dumps(parsed_calls)}</tool_call>",
                    })
                except Exception:
                    return None

            elif role == 'tool':
                try:
                    if tools is None:
                        return None
                    content = turn.get('content', '')
                    seq.append({
                        "role": "user",
                        "content": f"<tool_result>{content.strip()}</tool_result>",
                    })
                except Exception:
                    return None
        
        if len(seq) < 2:
            return None

        first_user_idx = None
        for i, msg in enumerate(seq):
            if msg['role'] == 'user':
                first_user_idx = i
                break
        
        if first_user_idx is None:
            return None

        messages = seq[:first_user_idx + 1]
        
        ground_tool_calls = []
        for msg in seq:
            if msg['role'] == 'assistant':
                tc_match = re.findall(r"<tool_call>(.*?)</tool_call>", msg['content'], re.DOTALL)
                if tc_match:
                    try:
                        ground_tool_calls.extend(json.loads(tc_match[0]))
                    except:
                        pass

        if not ground_tool_calls:
            return None

        return {
            "prompt": apply_chat_template(messages), # + [{"role": "assistant", "content": "```json\n"}]),
            "messages": messages,
            "def_tools": tools,
            "ground_tool_call": ground_tool_calls,
            "num_input_tools": len(tools),
            "scorer": partial(scorer, tools_ground=ground_tool_calls, def_tools=tools)
        }

    fc_dataset = load_dataset("LLM360/TxT360-3efforts", "agent", split='medium')
    print(f"Processing {len(fc_dataset)} samples...")
    train_ds = []
    count = 0
    for data in fc_dataset:
        result = mapper(data)
        if result is not None:
            train_ds.append(result)
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count} samples, {len(train_ds)} kept...")
    
    print(f"Dataset: {len(train_ds)} samples before filtering")

    if dedupe_ratio and dedupe_ratio < 1.0:
        try:
            from semhash import SemHash
            print("Running deduplication...")
            semhash = SemHash.from_records(train_ds, columns=['prompt'], use_ann=False)
            train_ds = semhash.self_deduplicate(threshold=dedupe_ratio)
            train_ds = train_ds.selected
        except Exception as e:
            print(f"Deduplication failed: {e}")

    # Optional: filter by prompt_token_len using rough estimate (char length / 4)
    if prompt_token_len > 0:
        print(f"Filtering by token length ({prompt_token_len})...")
        # Use rough estimate: ~4 chars per token
        rough_limit = prompt_token_len * 4
        train_ds = [x for x in train_ds if len(x['prompt']) <= rough_limit]
        print(f"  After rough filter: {len(train_ds)} samples")
        
        # For exact filtering, uncomment below (slow):
        # train_ds = [x for x in train_ds if len(tokenizer.encode(x['prompt'])) <= prompt_token_len]

    print("Input tool distribution:", np.bincount([x["num_input_tools"] for x in train_ds]))
    print(
        "Tool call distribution:",
        np.bincount([len(x["ground_tool_call"]) for x in train_ds]),
    )

    return train_ds


from data.grpo.verifiers import validate_format, tool_scorer, thinking_validate


def scorer(llm_gen, llm_judge, tools_ground, def_tools):
    # llm_gen = """```json\n""" + llm_gen
    valid_format = validate_format(llm_gen)
    if not valid_format:
        # print("Invalid format")
        return 0
    
    tool_score, tools_gen = tool_scorer(llm_gen, tools_ground, def_tools)
    return tool_score


if __name__ == '__main__':
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("weights/NanoAgent-135M")

    ds = txt360_toolcall(tokenizer, prompt_token_len=384)
    print("Dataset length:", len(ds))
    # print(tokenizer.apply_chat_template(ds[0]['prompt'], tokenize=False))
    print(ds[0]['prompt'])
    print(ds[0]['ground_tool_call'])