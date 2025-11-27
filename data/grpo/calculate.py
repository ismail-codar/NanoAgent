import random
import re
from functools import partial

def calculate_math(tokenizer, think=True):
    # rand = random.Random(seed)
    questions = []

    for i in range(0, 20):
        for j in range(0, 20):
            questions.append((f"What is {i} + {j}?", "+", i+j))
    
    for i in range(0, 20):
        for j in range(0, 20):
            questions.append((f"What is {i} - {j}?", "-", i-j))
    
    for i in range(-10, 10):
        questions.append((f"What is {i}**2?", "**", i**2))

    dataset = []
    for q in questions:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. You think inside <think> </think> tags before answering."},
            {"role": "user", "content": q[0]},
            {"role": "assistant", "content": f"<think>{random.choice(["Okay,", "Let's see,", "The user", "Let's think step by step"])}"}
        ]
        dataset.append({
            'prompt': tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'messages': messages,
            "def_tools": [],
            "ground_tool_call": q[-1],
            "num_input_tools": 0,
            "scorer": partial(scorer, answer=q[-1])
        })
    
    return dataset


def validate_format(text):
    """
    Validate if the text strictly follows the pattern:
    <think> ... </think><tool_call> ... </tool_call>

    Returns True if the string matches the pattern, False otherwise.
    """
    pattern = re.compile(
        r"^<think>.*?</think>\s.*?$", re.DOTALL
    )
    return bool(pattern.match(text)) and (text.count("</think>") == 1)



def thinking_scorer(llm_gen):
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think = pattern.findall(llm_gen)
    think = think[0]
    if len(think) > 16:
        return 1
    return -1


def scorer(llm_gen, answer):
    llm_gen = "<think>" + llm_gen
    if not validate_format(llm_gen):
        return -1
    if thinking_scorer(llm_gen) == -1:
        return -1
    if llm_gen.count("</") > 1:
        return -1
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    digits = re.findall(r'\d+', last_line)
    if digits:
        digit = int(digits[-1])
        if abs(digit - answer) <= 10:
            return (10 - abs(digit - answer)) / 10
        else:
            return 0
    return -1