#!/usr/bin/env python3
"""
Quick test script to verify NanoAgent inference with BFCL data works.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "weights/SmolLM2-135M-Instruct-nemotron-instruct-fc-instruct-sft"
BFCL_DATA_DIR = Path("/opt/homebrew/lib/python3.11/site-packages/bfcl_eval/data")


def load_model():
    """Load NanoAgent model and tokenizer."""
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


def generate_response(model, tokenizer, messages, max_new_tokens=256):
    """Generate response from NanoAgent."""
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
        print(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return ""


def main():
    model, tokenizer = load_model()
    
    # Load a single test case
    filepath = BFCL_DATA_DIR / "BFCL_v4_simple_python.json"
    with open(filepath, "r") as f:
        entry = json.loads(f.readline())
    
    print(f"\nTest entry ID: {entry.get('id')}")
    print(f"Functions: {json.dumps(entry.get('function', []), indent=2)[:500]}")
    print(f"Question: {entry.get('question')}")
    
    messages = prepare_messages(entry["question"], entry.get("function", []))
    print(f"\nPrepared messages: {json.dumps(messages, indent=2)[:1000]}")
    
    print("\nGenerating response...")
    response = generate_response(model, tokenizer, messages)
    print(f"\nResponse: {response[:500]}")
    
    # Try a second entry
    entry2 = json.loads(f.readline())
    messages2 = prepare_messages(entry2["question"], entry2.get("function", []))
    response2 = generate_response(model, tokenizer, messages2)
    print(f"\n\nEntry 2 Response: {response2[:500]}")


if __name__ == "__main__":
    main()
