#!/usr/bin/env python3
"""
NanoAgent BFCL Evaluation Wrapper

This script evaluates NanoAgent on the Berkeley Function Calling Leaderboard (BFCL)
using the model's native tool calling format.
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

BFCL_DATA_DIR = Path("/opt/homebrew/lib/python3.11/site-packages/bfcl_eval/data")
OUTPUT_DIR = Path("/Users/ohi/Documents/GitHub/NanoAgent/benchmarks/bfcl/result/nanoagent-nemotron")

MODEL_PATH = "weights/SmolLM2-135M-Instruct-nemotron-instruct-fc-instruct-sft"
# MODEL_PATH = "quwsarohi/NanoAgent-135M"

TEST_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "memory",
    "web_search",
]


def load_model():
    """Load NanoAgent model and tokenizer."""
    return load_model_path(MODEL_PATH)


def load_bfcl_data(category: str):
    """Load BFCL test data for a specific category."""
    filepath = BFCL_DATA_DIR / f"BFCL_v4_{category}.json"
    if not filepath.exists():
        print(f"Warning: {filepath} does not exist, skipping...")
        return []

    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_messages(question, functions, use_tools=True):
    """Prepare messages for NanoAgent inference."""
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


def generate_response(model, tokenizer, messages, max_new_tokens=384, prompt_lead="```json\n"):
    """Generate response from NanoAgent."""
    try:
        continue_final_message = prompt_lead is not None
        prompt = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": prompt_lead}] if prompt_lead is not None else [],
            tokenize=False,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
        )
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.to('mps'),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                min_p=0.2,
                repetition_penalty=1.05,
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        # print("--- PROMPT ---")
        # print(prompt)
        # print("--- RESPONSE ---")
        # print(response)
        if prompt_lead is not None:
            return prompt_lead + response
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""


def run_evaluation(model, tokenizer, categories=None):
    """Run evaluation on specified categories (or all if None)."""
    if categories is None:
        categories = TEST_CATEGORIES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for category in categories:
        print(f"\n{'='*50}")
        print(f"Processing category: {category}")
        print("="*50)
        
        output_file = OUTPUT_DIR / f"BFCL_v4_{category}_result.json"
        if os.path.exists(output_file):
            print("Skipping as benchmark on this category was already done.")
            continue

        data = load_bfcl_data(category)
        if not data:
            continue

        results = []
        total = len(data)
        tool_calls = 0

        for idx, entry in enumerate(tqdm(data, desc=f"Evaluating {category}")):
            entry_id = entry.get("id", f"{category}_{idx}")
            question = entry.get("question", [])
            functions = entry.get("function", [])

            messages = prepare_messages(question, functions)
            response = generate_response(model, tokenizer, messages)

            # Check if response contains a tool call
            has_tool_call = '```json' in response and '[{' in response
            if has_tool_call:
                tool_calls += 1

            result = {
                "id": entry_id,
                "result": [response] if response else [],
            }
            results.append(result)
            
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        tool_call_rate = (tool_calls / total * 100) if total > 0 else 0
        print(f"  Saved {len(results)} results to {output_file}")
        print(f"  Tool call rate: {tool_call_rate:.1f}% ({tool_calls}/{total})")

    print(f"\n{'='*50}")
    print("Evaluation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*50)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None, help="Specific category to run")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    args = parser.parse_args()
    
    model_path = args.model if args.model else MODEL_PATH
    
    model, tokenizer = load_model_path(model_path)
    
    if args.category:
        run_evaluation(model, tokenizer, [args.category])
    else:
        run_evaluation(model, tokenizer)


def load_model_path(model_path):
    """Load NanoAgent model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        # device_map='mps'
    ).to('mps')
    print("Model loaded successfully!")
    return model, tokenizer


if __name__ == "__main__":
    main()
