import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import sys
# sys.path.append("../")
from data.utils import json_toolcall_to_python, json_tooldef_to_python
from utils.tokenizer import TOOL_TEMPLATE_PY
    
model_name = "weights/SmolLM2-135M-nemotron-instruct-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16
)

def inference(messages, max_new_tokens=256, temperature=0.0, min_p=0.15, **kwargs):
    if isinstance(messages, list):
        continue_final_message = messages[-1]["role"] == "assistant"
        messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
        )
    inputs = tokenizer.encode(messages, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        min_p=min_p if temperature > 0 else None,
        temperature=temperature if temperature > 0 else None,
        **kwargs,
    )
    return tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)


print("Model loaded")
print("------------\n\n")

messages = []

while True:
    try:
        inp = input("User input: ")
        messages += [{"role": "user", "content": inp}]
        reply = inference(messages, max_new_tokens=384, min_p=0.2, temperature=0.1, repetition_penalty=1.05)
        messages += [{"role": "assistant", "content": reply}]
        print(reply)
        print()
    
    except KeyboardInterrupt:
        print("\nBye!")
        break
    

# messages = [
#     # {"role": "system", "content": TOOL_TEMPLATE},
#     {"role": "user", "content": "Define machine learning.\nAfter that, List 3 domains where ML can be applied in bullet points."},#Return your answer in JSON as the following format:\n```json\n{'concept': 'your_concept', 'definition': ...}\n```"},
#     # {"role": "user", "content": "Who is current the president of Bangladesh?"},
#     # {"role": "user", "content": "Write an email to my manager asking for parental leave extension."},
#     # {"role": "assistant", "content": "```python\n"}
#     # {"role": "assistant", "content": "The "}
#     # {"role": "user", "content": "Hi"},
# ]

# print("-"*30)
# print(inference(messages, max_new_tokens=384, ))#min_p=0.2, temperature=0.1, repetition_penalty=1.1)) # min_p=0.2, temperature=0.1, repetition_penalty=1.05


