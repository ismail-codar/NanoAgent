import random
import re
import reasoning_gym
from reasoning_gym import get_score_answer_fn
from functools import partial
from difflib import SequenceMatcher
from .verifiers import response_judge

JUDGE_TOKENS = 128

brainstorm_sentences = [
    "\nThink through the problem step by step, then present the final answer on the last line.",
    "\nDo your reasoning internally and only state the final conclusion at the end.",
    "\nAnalyze all possibilities first and write the final answer on the last line.",
    "\nBrainstorm thoroughly before giving the final answer in the last line.",
    "\nWork out the logic carefully, then provide the final answer at the end.",
    "\nConsider intermediate steps and output only the final answer on the last line.",
    "\nReason about the problem in detail, but show the final answer only at the end.",
    "\nEvaluate the problem step by step and conclude with the final answer on the last line.",
    "\nThink carefully through all steps, then write the final answer on the last line.",
    "\nPerform detailed reasoning first and place the final answer at the very end.",
    "\nInternally deliberate before responding, and give the final answer on the last line.",
    "\nComplete all analysis before presenting the final answer as the last line.",
    "",
    "",
    ""
]


reasoning_prefixes = [
    "Let's think step by step ",
    "I'll reason through this carefully ",
    "Let me work through the logic first ",
    "Let's brainstorm all possibilities ",
    "I'll carefully analyze the steps ",
    "Let me consider this internally ",
    "I'll reason about this in detail ",
    "Let's evaluate this one step at a time ",
    "I'll think this through carefully ",
    "Let me perform a detailed analysis ",
    "I'll deliberate on this internally ",
    "Let me complete the analysis first ",
    "",
    "",
    ""
]

def generate_think_string(no_think=False):
    if no_think:
        idx = random.choice(range(len(brainstorm_sentences[:-3])))
    else:
        idx = random.choice(range(len(brainstorm_sentences)))
    # return "", ""
    return brainstorm_sentences[idx], reasoning_prefixes[idx]


def number_sorting_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    last_line = last_line[last_line.find('['):]
    return score_fn(last_line, entry)


def number_sorting(tokenizer, size):
    dataset = reasoning_gym.create_dataset(
        name="number_sorting",   # task name
        min_numbers = 3,
        max_numbers = 10,
        min_decimals = 0,
        max_decimals = 1,
        min_value = -20,
        max_value = 20,
        seed = 42,
        size = size,
        # num_fewshot=1,
        # fewshot_as_multiturn=1
    )
    score_fn = get_score_answer_fn("number_sorting")
    instruction = ' Reply only with a name.'
    dataset_list = []

    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string()

        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': data['question']},
                    # {'role': 'assistant', 'content': llm_prefix}    
                ],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(number_sorting_parser, entry=data, score_fn=score_fn)
        })

    return dataset_list


def needle_haystack_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))
    if last_line:
        names = last_line[-1].strip().lower().split()
    else:
        return 0
    ans = entry['answer'].lower().strip()
    if ans in names:
        p = names.index(ans)
        return 1 / len(names[p:])
    return 0


def needle_haystack(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="needle_haystack",   # task name
        min_num_statements = 2,
        max_num_statements = 100,
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("needle_haystack")

    dataset_list = []
    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string()
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': data['question']} #+ user_suffix},
                    # {'role': 'assistant', 'content': llm_prefix}
                ],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(needle_haystack_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def syllogism_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    name = last_line.strip().split()[-1].capitalize().rstrip('.')
    score = score_fn(name, entry) * 0.5
    if score >= 0.5:
        judge_score = response_judge(entry['question'], response=llm_gen, n_tokens=JUDGE_TOKENS, ref_answer=entry['answer'])[1]
        return score + judge_score * 0.5
    return score


def syllogism(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="syllogism",   # task name
        allow_all = True,
        allow_no = True,
        allow_some = True,
        allow_some_not = True,
        invalid_ratio = 0.3,
        inversion_probability = 0.3,
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("syllogism")

    dataset_list = []
    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string()
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': data['question']},
                    # {'role': 'assistant', 'content': llm_prefix}
                ],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(syllogism_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def alice_in_wonderland_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    digits = re.findall(r'[+-]?\d+', last_line)
    if digits:
        score = score_fn(digits[-1], entry) * 0.5
        if score >= 0.5:
            judge_score = response_judge(entry['question'], response=llm_gen, n_tokens=JUDGE_TOKENS, ref_answer=entry['answer'])[1]
            return score + judge_score * 0.5
        return score
    return 0


def alice_in_wonderland(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="aiw",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("aiw")

    dataset_list = []
    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string()
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': data['question']},
                ],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(alice_in_wonderland_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def family_relationships_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))
    if len(last_line) == 0: return 0
    last_line = last_line[-1].strip().lower().split()
    answer = entry['answer'].lower().strip()
    if answer in last_line:
        p = last_line.index(answer)
        score = (1 / len(last_line[p:])) * 0.5
        if score >= 0.5:
            judge_score = response_judge(entry['question'], response=llm_gen, n_tokens=JUDGE_TOKENS, ref_answer=entry['answer'])[1]
            return score + judge_score * 0.5
        return score
    return 0


def family_relationships(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="family_relationships",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("family_relationships")

    dataset_list = []
    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string()
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': data['question']},
                    # {'role': 'assistant', 'content': llm_prefix}
                ],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(family_relationships_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def gsm_symbolic_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    digits = re.findall(r'[+-]?\d+', last_line)
    if digits:
        score = score_fn(digits[-1], entry) * 0.5
        if score >= 0.5:
            judge_score = response_judge(entry['question'], response=llm_gen, n_tokens=JUDGE_TOKENS, ref_answer=entry['answer'])[1]
            return score + judge_score * 0.5
        return score
    return 0


def gsm_symbolic(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="gsm_symbolic",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("gsm_symbolic")

    dataset_list = []
    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string()
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [{'role': 'user', 'content': data['question']}],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(gsm_symbolic_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def list_functions_parser(llm_gen, entry, score_fn):
    return score_fn(llm_gen.strip(), entry)


def list_functions(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="list_functions",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("list_functions")

    dataset_list = []
    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string()
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': data['question']} #+ user_suffix},
                    # {'role': 'assistant', 'content': llm_prefix}
                ],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(list_functions_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def codeio_parser(llm_gen, entry, score_fn):
    if '{' in llm_gen and '}' in llm_gen:
        l = llm_gen.find('{')
        r = llm_gen.find('}')
        return score_fn(llm_gen[l:r+1], entry)
    return score_fn(llm_gen, entry)


def codeio(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="codeio",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("codeio")

    dataset_list = []
    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string(no_think=True)
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': data['question']},
                    # {'role': 'assistant', 'content': llm_prefix}
                ],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(codeio_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


def chain_sum_parser(llm_gen, entry, score_fn):
    last_line = list(filter(lambda x: len(x.strip()) > 0, llm_gen.split('\n')))[-1].strip()
    digits = re.findall(r'[+-]?\d+', last_line)
    if digits:
        digit = float(digits[-1])
        ans = float(entry['answer'])
        margin = 10
        if abs(digit - ans) <= margin:
            score = ((margin - abs(digit - ans)) / margin) * 0.5
            if score >= 0.5:
                judge_score = response_judge(entry['question'], response=llm_gen, n_tokens=JUDGE_TOKENS, ref_answer=entry['answer'])[1]
                return score + judge_score * 0.5
            return score
    return 0


def chain_sum(tokenizer, size=500, prompt_token_len=None):
    dataset = reasoning_gym.create_dataset(
        name="chain_sum",   # task name
        seed = 42,
        size = size,
    )
    score_fn = get_score_answer_fn("chain_sum")

    dataset_list = []
    for data in dataset:
        # user_suffix, llm_prefix = generate_think_string(no_think=True)
        dataset_list.append({
            'prompt': tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': data['question']},
                    # {'role': 'assistant', 'content': llm_prefix}
                ],
                add_generation_prompt=True,
                tokenize=False,
                continue_final_message=False,
            ),
            'answer': data['answer'],
            'scorer': partial(chain_sum_parser, entry=data, score_fn=score_fn)
        })

    if prompt_token_len:
        dataset_list = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset_list))

    return dataset_list


if __name__ == '__main__':
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")

    # ds = alice_in_wonderland(tokenizer=tokenizer)
    ds = family_relationships(tokenizer)
    print(ds[-1]['prompt'])
    answer = "THINKING...\nFinal Answer: " + ds[-1]['answer']
    print(answer)
    print(ds[-1]['scorer'](answer))