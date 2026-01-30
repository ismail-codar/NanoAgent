from functools import partial
from datasets import load_dataset
from .sandbox import DockerSandbox
from .verifiers import get_llm_response, response_judge


docker_sandbox = None
JUDGE_TOKENS = 512

def scorer(response, llm_judge, eval_func, question):
    global docker_sandbox
    if docker_sandbox is None:
        docker_sandbox = DockerSandbox()

    if len(response.strip()) <= 32:
        return 0 

    results = []
    for ef in eval_func:
        code_string = f"{ef}\n\nresponse = '{response}'\n\nprint(evaluate(response))"
        try:
            result = docker_sandbox.run_code(code_string).lower().strip() == 'true'
        except Exception as E:
            print("Exception:", E)
            result = False
        results.append(result)

    final_result = sum([float(r) for r in results]) / len(results)
    if final_result > 0 and llm_judge:
        score = response_judge(question=question, response=response, n_tokens=JUDGE_TOKENS, strict_level=2)[1]
        score = (score * 3) / 2
        final_result *= score
    return final_result


def autoif_ds(tokenizer, prompt_token_len, n_instructions=None):
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
    if n_instructions:
        dataset = list(filter(lambda d: len(set(d['eval_funcs'])) <= n_instructions, dataset))
    for i in range(len(dataset)):
        dataset[i]['scorer'] = partial(scorer, eval_func=dataset[i]['eval_funcs'], question=dataset[i]['messages'][-1]['content'])
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