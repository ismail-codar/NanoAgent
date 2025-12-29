from functools import partial
from datasets import load_dataset
from .sandbox import DockerSandbox


docker_sandbox = DockerSandbox()

def scorer(eval_func, response):
    code_string = f"response = '{response}'\n\n{eval_func[0]}"
    return docker_sandbox.run_code(code_string).lower().strip() == 'true'

def autoif(tokenizer):
    dataset = load_dataset("Post-training-Data-Flywheel/AutoIF-instruct-61k-with-funcs")['train']
    dataset = dataset.map(lambda x: {'eval_funcs': list(set(x['eval_funcs']))})
    dataset = dataset.map(lambda x: {
        'prompt': tokenizer.apply_chat_template(
            x['messages'][:-1],
            add_generation_prompt=True,
            tokenize=False,
            continue_final_message=False,
    )})
    dataset = dataset.remove_columns(['system', 'tools', 'conversation_id'])
    dataset = [d for d in dataset]
    for i in range(len(dataset)):
        dataset[i]['scorer'] = partial(scorer, eval_func=dataset[i]['eval_funcs'])
    return dataset


if __name__ == '__main__':
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")

    ds = autoif(tokenizer)
    print(len(ds))
    print(json.dumps(ds[10]))