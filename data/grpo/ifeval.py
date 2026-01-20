# See https://github.com/allenai/open-instruct/blob/c9b10651d6be32bd7bb61669f8ba3d09127bfb04/open_instruct/ground_truth_utils.py#L281


import ast
import re
import json
from functools import partial
from data.grpo.IFEvalG import instructions_registry
from abc import ABC, abstractmethod
from typing import Any
import asyncio
import random
from datasets import load_dataset
from data.grpo.verifiers import response_judge

KSHOT_EXAMLPES = [
    [
        {'role': 'user', 'content': 'Greet me in lowercase with only one exclamation mark'},
        {'role': 'assistant', 'content': 'hi! how can i help you?'}
    ],
    [
        {'role': 'user', 'content': 'What is 2+3= Reply the number in two decimal places'},
        {'role': 'assistant', 'content': '5.00'}
    ],
    [
        {'role': 'user', 'content': 'Respond with yes or no only: Is water wet?'},
        {'role': 'assistant', 'content': 'yes'}
    ],
    [
        {'role': 'user', 'content': 'Write the word HELLO in lowercase'},
        {'role': 'assistant', 'content': 'hello'}
    ],
    [
        {'role': 'user', 'content': 'What is the capital of France? Answer in one word.'},
        {'role': 'assistant', 'content': 'Paris'}
    ],
    [], [], []
]


def filter_non_english(text: str) -> bool:
    """
    Detects if the given text contains characters that are not basic English.
    English includes A-Z, a-z, 0-9, and basic punctuation.
    """
    # Match anything that is not basic English letters, numbers or punctuation
    return not bool(re.search(r"[^\x00-\x7F]", text))


def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


class VerifierFunction(ABC):
    """
    Base class for all verifier functions that evaluate model predictions against ground truth.

    Each verifier function takes a prediction and compares it to a ground truth label,
    returning a VerificationResult with a score between 0.0 and 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0, verifier_config = None) -> None:
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config


    @abstractmethod
    def __call__(self, prediction: str, label: str | dict, question: str):
        """
        Evaluate the given prediction against the ground truth (or constraint).

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query

        Returns:
            VerificationResult
        """

    async def async_call(self, prediction: str, label: str | dict, question: str):
        """
        Asynchronous version of __call__. By default, it runs the synchronous __call__ in a thread pool.
        Subclasses can override this method for truly asynchronous implementation.

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query.

        Returns:
            VerificationResult
        """
        # Run the synchronous __call__ in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.__call__(prediction, label, question))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"


class IFEvalVerifier(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint(s) are a list of constraint ids.
    This list is found under the key "instruction_id" in the ground_truth dict.
    """

    def __init__(self) -> None:
        super().__init__("ifeval", weight=1.0)

    def __call__(self, prediction: str, label: str | dict, question: str) -> float:
        instruction_dict = instructions_registry.INSTRUCTION_DICT
        constraint_dict = ast.literal_eval(label)
        constraint_dict = constraint_dict[0]
        if isinstance(constraint_dict, str):
            constraint_dict = json.loads(constraint_dict)
        answer = remove_thinking_section(prediction)
        instruction_keys = constraint_dict["instruction_id"]
        args_list = constraint_dict["kwargs"]
        rewards = []
        if len(prediction) == 0 or len(answer) == 0:
            # logger.warning("Empty prediction received for IFEvalVerifier.")
            return 0
        
        for instruction_key, args in zip(instruction_keys, args_list):
            if args is None:
                args = {}
            args = {k: v for k, v in args.items() if v is not None}
            instruction_cls = instruction_dict[instruction_key]
            instruction_instance = instruction_cls(instruction_key)
            instruction_instance.build_description(**args)
            if prediction.strip() and instruction_instance.check_following(answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        score = sum(rewards) / len(rewards)
        if score <= 0:
            return score
        
        # LLM reward hacking the answer
        if len(prediction.strip().split()) <= 16:
            return 0
        
        n_tries = 1
        judge_scores = []
        for _ in range(n_tries):
            judge_resp, judge_score = response_judge(question=question, response=prediction, n_tokens=512, strict_level=2)
            judge_scores.append(judge_score)
        # print("Judge Scores:", judge_scores)
        return score * (sum(judge_scores) / len(judge_scores))
    

scorer = IFEvalVerifier()

def ifeval_ds(tokenizer, prompt_token_len, n_instructions=None, kshot=False):
    dataset = load_dataset("allenai/Dolci-RL-Zero-IF-7B")['train']
    # dataset = dataset.map(lambda x: {'eval_funcs': list(set(x['eval_funcs']))})
    dataset = dataset.map(lambda x: {
        'question': x['prompt'].strip().lstrip('user:').strip(),
        'prompt': tokenizer.apply_chat_template(
            (random.choice(KSHOT_EXAMLPES) if kshot else []) + [{'role': 'user', 'content': x['prompt'].strip().lstrip('user:').strip()}],
            add_generation_prompt=True,
            tokenize=False,
            continue_final_message=False,
    )})
    dataset = dataset.filter(lambda d: filter_non_english(d['question']))
    dataset = dataset.remove_columns(['constraint'])
    dataset = [d for d in dataset]
    for i in range(len(dataset)):
        dataset[i]['scorer'] = partial(scorer, label=dataset[i]['ground_truth'][0], question=dataset[i]['question'])

    if n_instructions:
        dataset = list(filter(lambda x: len(ast.literal_eval(x['ground_truth'][0])[0]['instruction_id']) <= n_instructions, dataset))
    dataset = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset))
    return dataset


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")
    # constraint = [{
    #     'instruction_id': ['keywords:forbidden_words', 'length_constraints:number_paragraphs', 'count:counting_composition'], 
    #     'kwargs': [{'forbidden_words': ['commission', 'population', 'road', 'stuff']}, {'num_paragraphs': 6}, {'n_sent': 2, 'n_words': 2}]
    # }]

    # print(ver("This is a sentence", str(constraint)))
    dataset = ifeval_ds(tokenizer, 256)
    print(dataset[0])
