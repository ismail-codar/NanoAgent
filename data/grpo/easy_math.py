import random
import re
from functools import partial
from .verifiers import get_llm_response, response_judge


def scorer(llm_gen, question, answer, diff=5):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    digits = re.findall(r'\d+', last_line)
    if digits:
        digit = int(digits[-1])
        if abs(digit - answer) <= diff:
            score = ((diff - abs(digit - answer)) / diff)
            judge_score = response_judge(question=question, response=llm_gen, n_tokens=128)[1]
            return score * 0.5 + judge_score * 0.5
    return 0


def easymath(tokenizer, think=False):
    questions = []
    answers = []

    for _ in range(1_000):
        a = random.randint(0, 50)
        b = random.randint(0, 50)
        questions.append(f"Solve the math problem:\n{a} + {b} = ?")
        answers.append(a+b)
    
    for _ in range(1_000):
        a = random.randint(0, 50)
        b = random.randint(0, 50)
        questions.append(f"Solve the math problem:\n{a} - {b} = ?")
        answers.append(a-b)
    
    for _ in range(1_000):
        b = random.randint(1, 10)
        a = random.randint(0, 10) * b
        questions.append(f"Solve the math problem:\n{a} / {b} = ?")
        answers.append(a // b)

    for _ in range(1_000):
        b = random.randint(0, 10)
        a = random.randint(0, 10)
        questions.append(f"Solve the math problem:\n{a} * {b} = ?")
        answers.append(a * b)
    
    dataset = []
    for (ques, ans) in zip(questions, answers):
        if think:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. You think inside <think> </think> tags before answering."},
                {"role": "user", "content": ques},
                {"role": "assistant", "content": f"<think>{random.choice(["Okay,", "Let's see,", "The user", "Let's think step by step"])}"}
            ]
        else:
            messages = [
                {"role": "user", "content": ques + "\nThink before giving answer."},
                # {"role": "assistant", "content": random.choice(["Okay,", "Let's see,", "The user", "Let's think step by step"])
            ]

        dataset.append({
            'prompt': tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'messages': messages,
            "answer": ans,
            "scorer": partial(scorer, question=ques, answer=ans)
        })
    
    return dataset


# def validate_format(text):
#     """
#     Validate if the text strictly follows the pattern:
#     <think> ... </think><tool_call> ... </tool_call>

#     Returns True if the string matches the pattern, False otherwise.
#     """
#     pattern = re.compile(
#         r"^<think>.*?</think>\s.*?$", re.DOTALL
#     )
#     return bool(pattern.match(text)) and (text.count("</think>") == 1)



# def thinking_scorer(llm_gen):
#     pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
#     think = pattern.findall(llm_gen)
#     think = think[0]
#     if len(think) > 16:
#         return 1
#     return -1


# def scorer(llm_gen, answer):
#     llm_gen = "<think>" + llm_gen
#     if not validate_format(llm_gen):
#         return -1
#     if thinking_scorer(llm_gen) == -1:
#         return -1
#     if llm_gen.count("</") > 1:
#         return -1
#     last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
#     digits = re.findall(r'\d+', last_line)
#     if digits:
#         digit = int(digits[-1])
#         if abs(digit - answer) <= 10:
#             return (10 - abs(digit - answer)) / 10
#         else:
#             return 0
#     return -1

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")
    data = calculate_math(tokenizer)

    print(data[0])
    print(data[0]['scorer'](f"The answer is {data[0]['answer']}"))
    print(data[0]['scorer'](f"The answer is {data[0]['answer']+99}"))
    print(data[0]['scorer'](f"The answer is {data[0]['answer']+5}"))