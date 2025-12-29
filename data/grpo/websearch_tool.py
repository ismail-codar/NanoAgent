import json
import re
from ast import literal_eval
from datasets import load_dataset
from utils.tokenizer import TOOL_TEMPLATE
from data.utils import tool_shuffle
from functools import partial
from .verifiers import validate_format, tool_scorer, thinking_validate

def tool_calling_traces(tokenizer):
    visit_webpage = {
        "name": "visit_webpage",
        "description": "Fetches and displays the textual content of a webpage (converted to Markdown) from a given URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the webpage to retrieve.",
                }
            },
            "required": ["url"],
        },
    }

    def mapper(data):
        tool_strs = data["chat_template_kwargs"]["xml_tools"][0].split("\n")
        tools = [json.loads(tool_str)["function"] for tool_str in tool_strs] + [
            visit_webpage
        ]
        p = -1
        for idx, t in enumerate(tools):
            if t['name'] == 'wikipedia_search':
                p = idx
                break
        del tools[p]
        all_messages = []
        tool_names = [tool["name"] for tool in tools]
        tools_json = tool_shuffle(tools)
        tool_calls = None
        messages = [
            {
                "role": "system",
                "content": TOOL_TEMPLATE.format(tools=tools_json),
            }
        ]
        # if "final_answer" in tool_names:
        #     messages[-1]["content"] += (
        #         " Always call 'final_answer' when answer is found."
        #     )
        for turn in data["messages"]:
            if turn["role"] == "assistant":
                think = re.findall(r"<think>(.*?)</think>", turn["content"], re.DOTALL)[0]
                tool_calls = re.findall(
                    r"<tool_call>(.*?)</tool_call>", turn["content"], re.DOTALL
                )[0]
                tool_calls = literal_eval(tool_calls)
                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]
                for tc in tool_calls:
                    if tc["name"] not in tool_names:
                        print(f"{tc['name']} not present in tool def, {tc}")

                # Append this as a message
                all_messages.append({
                    "prompt": tokenizer.apply_chat_template(
                        messages + [{"role": "assistant", "content": "<tool_call>"}],
                        add_generation_prompt=False,
                        tokenize=False,
                        continue_final_message=True,
                    ),
                    "messages": messages,
                    "def_tools": tools,
                    "ground_tool_call": tool_calls,
                    "num_input_tools": len(tools),
                    "category": tool_calls[0]['name'],
                    "scorer": partial(scorer, tools_ground=tool_calls, def_tools=tools)
                })

                messages.append({
                    "role": "assistant",
                    "content": f"<tool_call>{json.dumps(tool_calls)}</tool_call>",
                })

            elif turn["role"] == "tool":
                messages.append({
                    "role": "user",
                    "content": f"<tool_result>{turn['content'].strip()}</tool_result>",
                })
            elif turn["role"] == "user":
                messages.append({"role": "user", "content": turn["content"]})
            else:
                raise NotImplementedError(f"Role: {turn['role']}")
            
        return all_messages

    ds = load_dataset(
        "parquet",
        data_files="data/datasets/smolagents_toolcalling_traces_think-00000-of-00001.parquet",
    )["train"]
    # ds = list(map(mapper, ds))
    new_dataset = []
    for data in ds:
        new_dataset.extend(mapper(data))
    # ds = ds.remove_columns(["chat_template_kwargs"])
    # ds = [d for d in ds]
    return new_dataset


def scorer(llm_gen, tools_ground, def_tools):
    # Adding think tag (prefilled in dataset)
    # Tool score
    llm_gen = "<tool_call>" + llm_gen
    # print("LLM_GEN:", llm_gen)
    tool_score, tools_gen = tool_scorer(llm_gen, tools_ground, def_tools)
    # if tool_score <= 0:
        # return -1
    return tool_score


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")
    def total_tokens(data):
        return len(
            tokenizer.encode(
                data,
            )  # add_generation_prompt=True)
        )

    ds = tool_calling_traces(tokenizer)
    ds = list(filter(lambda x: 512 <= total_tokens(x['prompt']) <= 1536, ds))
    from collections import Counter
    fc_names = []
    for d in ds:
        fc_names.append(d['ground_tool_call'][0]['name'])
    print("Dataset length:", len(ds))
    print(Counter(fc_names))