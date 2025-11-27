import ast
from ast import literal_eval
import json
from copy import deepcopy
from functools import partial

from data.utils import tool_shuffle
from utils.tokenizer import TOOL_TEMPLATE
from utils.webtool import tool_call_extract
from data.utils import THINK_STRINGS
from data.grpo.salseforce_tool import tool_scorer

# Example usage:
# print(func_call_to_json('coffee_shop.find_nearby("San Francisco", amenities="Wi-Fi", rating=5)'))


def gorilla_openfun(tokenizer):
    def func_call_to_json(call_str):
        node = ast.parse(call_str, mode='eval').body
        if not isinstance(node, ast.Call):
            raise ValueError("Input must be a function call.")
        
        # Get full function name (supports dotted attributes)
        def get_full_name(func):
            if isinstance(func, ast.Attribute):
                parts = []
                cur = func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                return ".".join(reversed(parts))
            elif isinstance(func, ast.Name):
                return func.id
            else:
                raise ValueError("Unsupported function type.")
        
        # Safe evaluator for argument nodes
        def safe_eval(node):
            if isinstance(node, ast.Constant):  # For Python 3.8+
                return node.value
            elif isinstance(node, ast.Str):  # Older Python versions
                return node.s
            elif isinstance(node, ast.Name):
                # Treat undefined variable as a string placeholder
                return node.id
            elif isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.List):
                return [safe_eval(elt) for elt in node.elts]
            elif isinstance(node, ast.Dict):
                return {safe_eval(k): safe_eval(v) for k, v in zip(node.keys, node.values)}
            elif isinstance(node, ast.Tuple):
                return tuple(safe_eval(elt) for elt in node.elts)
            elif isinstance(node, ast.Attribute):
                # Return dotted name like `module.object`
                parts = []
                cur = node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                return ".".join(reversed(parts))
            else:
                # Fallback: return unparsed code for complex expressions
                return ast.unparse(node) if hasattr(ast, "unparse") else str(node)

        func_name = get_full_name(node.func)

        # Extract arguments safely
        args = [safe_eval(arg) for arg in node.args]
        kwargs = {kw.arg: safe_eval(kw.value) for kw in node.keywords}

        # Build the JSON structure
        json_obj = {
            "name": func_name,
            "arguments": {
                "args": args,
                "kwargs": kwargs
            }
        }
        # return json.dumps(json_obj, indent=2)
        return json_obj
    
    gorilla = []
    with open("/Users/ohi/Documents/GitHub/EdgeAgent/notebooks/gorilla_openfunctions_v1_train.json", "r") as f:
        for line in f:
            gorilla.append(json.loads(line))
    with open("/Users/ohi/Documents/GitHub/EdgeAgent/notebooks/gorilla.json", "r") as f:
        gorilla.extend(json.load(f))

    train_data = []

    for i in range(len(gorilla)):
        try:
            inps = gorilla[i]['Output']
            if isinstance(inps, list):
                outs = [func_call_to_json(inp) for inp in inps]
            else:
                outs = [func_call_to_json(inps)]
            
            tools = []
            for f in gorilla[i]['Functions']:
                f = literal_eval(f.strip())
                f['name'] = deepcopy(f['api_name'])
                del f['api_name']
                tools.append(f)
            tool_defs = {t['name']: t for t in tools}

            # Checks
            for o in outs:
                if o['name'] not in tool_defs:
                    raise ValueError

            for i, c in enumerate(outs):
                if len(c['arguments']['args']) > 0:
                    raise ValueError

            tools_called = []
            for c in outs:
                tools_called.append({'name': c['name'], 'arguments': c['arguments']['kwargs']})

            # train_data.append({
            #     'messages': [
            #         {'role': 'system', 'content': TOOL_TEMPLATE.format(tools=tool_shuffle(tools))},
            #         {'role': 'user', 'content': gorilla[i]['Instruction']},
            #         {'role': 'assistant', 'content': f"<tool_call>{tools_called}</tool_call>"}
            #     ],
            #     'source': 'https://github.com/ShishirPatil/gorilla'
            # })

            messages = [
                {'role': 'system', 'content': TOOL_TEMPLATE.format(tools=tool_shuffle(tools))},
                {'role': 'user', 'content': gorilla[i]['Instruction']}
            ]

            train_data.append({
                "prompt": tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    continue_final_message=False,
                ),
                "messages": messages,
                "def_tools": tools,
                "ground_tool_call": tools_called,
                "num_input_tools": len(tools),
                "scorer": partial(scorer, tools_ground=tools_called, def_tools=tools)
            })

        except:
            pass

    return train_data

# data = gorilla_openfun(TOOL_TEMPLATE)


def scorer(llm_gen, tools_ground, def_tools):
    # Adding think tag (prefilled in dataset)
    # Tool score
    tool_score, tools_gen = tool_scorer(llm_gen, tools_ground)
    if tool_score <= 0:
        return -1
    return tool_score