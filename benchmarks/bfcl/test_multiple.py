#!/usr/bin/env python3
"""
Test script to see multiple BFCL responses from NanoAgent.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "weights/SmolLM2-135M-Instruct-nemotron-instruct-fc-instruct-sft"
BFCL_DATA_DIR = Path("/opt/homebrew/lib/python3.11/site-packages/bfcl_eval/data")


def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully!")
    return model, tokenizer


def prepare_messages(question, functions, use_tools=True):
    messages = []

    if use_tools and functions:
        tools_json = json.dumps(functions, indent=2)
        system_content = f"""You are a helpful AI assistant. You have a set of possible tools that you can execute to retrieve information or to perform specific actions. You can execute zero or more tools to answer user question.

Here are the list of tools that you have access to:
```json
{tools_json}
```

Only execute tools from above. Follow the below JSON signature to execute tools:
```json
[{{"name": "tool_name", "arguments": {{"arg1": "val1", ...}}}}, ...]
```"""
        messages.append({"role": "system", "content": system_content})

    for msg_list in question:
        for msg in msg_list:
            messages.append({"role": msg["role"], "content": msg["content"]})

    return messages


def generate_response(model, tokenizer, messages, max_new_tokens=256):
    try:
        continue_final_message = messages[-1].get("role") == "assistant"
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
        )
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                min_p=0.2,
                repetition_penalty=1.05,
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error: {e}")
        return ""


def main():
    model, tokenizer = load_model()
    
    filepath = BFCL_DATA_DIR / "BFCL_v4_simple_python.json"
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    print(f"\nLoaded {len(lines)} test cases\n")
    
    # Test first 5 entries
    for i in range(min(5, len(lines))):
        entry = json.loads(lines[i])
        
        messages = prepare_messages(entry["question"], entry.get("function", []))
        
        print(f"--- Test {i+1}: {entry.get('id')} ---")
        print(f"Question: {entry['question'][0][0]['content'][:100]}...")
        print(f"Function: {entry['function'][0]['name'] if entry.get('function') else 'None'}")
        
        response = generate_response(model, tokenizer, messages)
        
        # Check if response contains tool call
        has_tool_call = '```json' in response and '[{' in response
        print(f"Response (first 300 chars): {response[:300]}...")
        print(f"Contains tool call: {has_tool_call}")
        print()


if __name__ == "__main__":
    main()
