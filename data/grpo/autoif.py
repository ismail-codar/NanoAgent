from functools import partial
from datasets import load_dataset
from .sandbox import DockerSandbox


docker_sandbox = DockerSandbox()

def scorer(response, eval_func):
    results = []
    for ef in eval_func:
        code_string = f"{ef}\n\nresponse = '{response}'\n\nprint(evaluate(response))"
        result = docker_sandbox.run_code(code_string).lower().strip() == 'true'
        results.append(result)
    return all(results)


def autoif_ds(tokenizer, prompt_token_len):
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
    dataset = list(filter(lambda x: len(tokenizer.encode(x['prompt'])) <= prompt_token_len, dataset))
    return dataset


if __name__ == '__main__':
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")

    ds = autoif(tokenizer, 256)
    print(len(ds))
    # print(json.dumps(ds[10]))

    print(ds[10]['eval_funcs'])

    print(ds[10]['scorer'](response=ds[10]['messages'][-1]['content']))
    print(ds[120]['scorer'](response=ds[10]['messages'][-1]['content']))