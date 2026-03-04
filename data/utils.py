import ast
import json
import random
import re
import textwrap
from ast import literal_eval

# Prompts to generate reasoning chain
THINK_STRINGS = [
    "You must think step-by-step inside <think> </think> tags before answering.",
    "Start by reasoning through the problem inside the <think> </think> tags, then answer.",
    "Put your full reasoning in <think> </think> tags before arriving at an answer.",
    "Always explain your logic step-by-step inside <think> </think> tags before responding.",
    "Your answer must follow a step-by-step reasoning written inside <think> </think> tags.",
    "Provide your thought process inside <think> </think> tags before writing the final answer.",
    "Use the <think> </think> tags to write your reasoning step-by-step before answering.",
    "Think through the problem inside <think> </think> tags before you respond.",
    "Make sure to include your step-by-step thinking inside the 'think' tags prior to answering.",
    "Please reason step-by-step within the <think> </think> tags before giving your answer.",
    "Think inside 'think' XML tags and then provide final answer",
    "Think step-by-step inside 'think' tags.",
    "Think inside 'think' tags before generating final answer.",
]

def tool_parse(tool_call: str):
    """
    Parses tool call in two different formats:
    {'function_name': 'fun1', 'arguments': {...}}
    {"function_name": "fun1", "arguments": {...}}
    """
    try:
        return literal_eval(tool_call)
    except:
        pass
    try:
        return json.loads(tool_call)
    except:
        pass
    return None

def tool_shuffle(tool):
    """
    Randomly shuffles the input tool and returns a randomly selected string representation.

    Parameters:
        tool (list or any): The input to be shuffled. If not a list, it will be converted into a list.
                            The elements of the list must not be of type `str`.

    Returns:
        str: A randomly selected string representation of the shuffled tool. The representation
             could be a simple string conversion or a JSON-formatted string with varying indentation.

    Raises:
        AssertionError: If the first element of the input list is of type `str`.
    """
    if not isinstance(tool, list):
        tool = [tool]
    if tool:
        assert not isinstance(tool[0], str), (
            f"Tool type should not be a str: type-{type(tool[0])}"
        )
    random.shuffle(tool)
    reps = [str(tool)]
    for c in [None, 1, 2, 3, 4]:
        reps.append(json.dumps(tool, indent=c))
    return random.choice(reps)


def extract_tag(input_str, tag):
    tool_def = re.findall(f"<{tag}>(.*?)</{tag}>", input_str, re.DOTALL)
    tool_def = map(str.strip, tool_def)
    tool_def = filter(lambda x: len(x) > 0, tool_def)
    return list(tool_def)


def remove_code_mentions(text: str, remove_python=False) -> bool:
    langs = [
        "JavaScript",
        "Java",
        "C",
        "C++",
        "C#",
        "TypeScript",
        "Go",
        "Rust",
        "Kotlin",
        "Swift",
        "PHP",
        "Ruby",
        "Shell",
        "SQL",
        "Dart",
        "R",
        "MATLAB",
        "Objective-C",
        "Scala",
        "Perl",
    ]

    if remove_python:
        langs.append("Python")

    for l in langs:
        if l.lower() in text.lower().split():
            return False
    return True


def code_markdown_filter(markdown: str, remove_python=False) -> bool:
    # Find all fenced code blocks like ```lang\ncode\n```
    code_blocks = re.findall(r"```(\w*)\n.*?```", markdown, flags=re.DOTALL)
    filters = ["markdown", "json"]
    if not remove_python:
        filters.append("python")
    for lang in code_blocks:
        if lang.strip().lower() not in filters:
            return False
    return True


common_libraries = set(
    [
        "os",
        "sys",
        "math",
        "random",
        "datetime",
        "time",
        "re",
        "json",
        "csv",
        "itertools",
        "collections",
        "functools",
        "typing",
        "logging",
        "subprocess",
        "pathlib",
        "shutil",
        "copy",
        "threading",
        "multiprocessing",
        "http",
        "urllib",
        "socket",
        "argparse",
        "dataclasses",
        "heapq",
        "traceback",
        "pytest",
        "pprint",
        "decimal",
        "fractions",
        "statistics",
        "enum",
        "inspect",
        "warnings",
        "email",
        "sqlite3",
        "requests",
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
    ]
)


def get_imported_packages(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()  # Skip code blocks with syntax errors

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def extract_python_code_blocks(markdown_text):
    # Find all python code blocks
    code_blocks = re.findall(r"```python(.*?)```", markdown_text, re.DOTALL)
    # Dedent each block to remove markdown-based indentation
    return [textwrap.dedent(block.strip()) for block in code_blocks]


def filter_python_package(conv):
    def extract_imports_from_markdown(markdown_text):
        code_blocks = extract_python_code_blocks(markdown_text)
        all_imports = set()

        for code in code_blocks:
            all_imports.update(get_imported_packages(code))

        return sorted(all_imports)

    packs = extract_imports_from_markdown(conv)
    for p in packs:
        if p not in common_libraries:
            return False
    return True


def short_code(inp_str):
    codes = extract_python_code_blocks(inp_str)
    for code in codes:
        lines = [l.strip() for l in code.split("\n")]
        n_lines = 0
        for l in lines:
            if not l.startswith("#"):
                n_lines += 1
        if n_lines > 25:
            return False

    return True


def dedent_markdown_python_code(markdown_text):
    def dedent_code_block(match):
        code = match.group(1)
        dedented = textwrap.dedent(code.strip('\n'))
        return f"```python\n{dedented.strip()}\n```"

    # Substitute each code block with a dedented version
    cleaned_markdown = re.sub(r' *```python(.*?)```', dedent_code_block, markdown_text, flags=re.DOTALL)
    return cleaned_markdown


def filter_by_resp_len(ds, resp_lim:int, messages_key='messages'):
    def map_fn(data):
        new_messages = []
        cache_messages = []
        for turn in data["_"+messages_key]:
            if turn['role'] == 'assistant':
                if len(turn['content']) > resp_lim:
                    break
                new_messages.extend(cache_messages + [turn])
                cache_messages = []
            else:
                cache_messages.append(turn)
        return new_messages
    
    
    ds = ds.rename_column(messages_key, "_"+messages_key)
    ds = ds.map(lambda d: {'messages': map_fn(d), **d})
    ds = ds.filter(lambda d: len(d['messages']) > 0)
    ds = ds.remove_columns(['_'+messages_key])
    
    return ds


def filter_non_english(text: str) -> bool:
    """
    Detects if the given text contains characters that are not basic English.
    English includes A-Z, a-z, 0-9, and basic punctuation.
    """
    # Match anything that is not basic English letters, numbers or punctuation
    return not bool(re.search(r"[^\x00-\x7F]", text))


def pack_data(
    dataset, ctx_len=1024, return_list=False, segment_size=256, sort=False, report=True
):
    from datasets import Dataset
    new_dataset = []
    tot_buckets = (ctx_len // segment_size) + 2
    ctx_bucket = [set() for _ in range(tot_buckets)]
    assert segment_size > 1

    if sort:
        dataset = dataset.sort("ctx_len")

    for idx, data in enumerate(dataset):
        prev_loc_found = False
        # Iterate over bucket size
        # Bucket 0 -> [0, segment_size)
        # Bucket 1 -> [segment_size, 2*segment_size)
        for bidx in range(tot_buckets):
            # Bucket index indicates remaining ctx_len
            min_rem_ctx_len = (bidx + 1) * segment_size
            # If remaining_len is greater than the content ctx len and a content exists
            # Truncation happens due to data["ctx_len"] // 2
            if min_rem_ctx_len >= (data["ctx_len"] // 2) and ctx_bucket[bidx]:
                # Add the content in the existing data
                data_idx = next(iter(ctx_bucket[bidx]))
                new_dataset[data_idx]["messages"].append(data["messages"])
                new_dataset[data_idx]["source"].append(data["source"])
                new_dataset[data_idx]["ctx_len"] += data["ctx_len"]
                ctx_bucket[bidx].remove(data_idx)
                prev_loc_found = True
                # New remaining ctx_len
                new_rem_ctx_len = max(ctx_len - new_dataset[data_idx]["ctx_len"], 0)
                if new_rem_ctx_len < 16: break
                new_bidx = (new_rem_ctx_len // segment_size) + 1
                ctx_bucket[new_bidx].add(data_idx)
                break

        if not prev_loc_found:
            new_dataset.append(
                {
                    "messages": [data["messages"]],
                    "source": [data["source"]],
                    "ctx_len": data["ctx_len"],
                }
            )
            rem_ctx_idx = (max(ctx_len - new_dataset[-1]["ctx_len"], 0) // segment_size) + 1
            ctx_bucket[rem_ctx_idx].add(len(new_dataset) - 1)

    if report:
        # new_dataset = pack_data(new_dataset, ctx_len, sort=True, report=False)
        NON_PAD_TOKS = 0
        TRUNCATED_TOKS = 0
        PAD_TOKS = 0
        TOT_TOKS = 0

        for idx, data in enumerate(new_dataset):
            TRUNCATED_TOKS += max(data["ctx_len"] - ctx_len, 0)
            NON_PAD_TOKS += data["ctx_len"]
            PAD_TOKS += max(ctx_len - data['ctx_len'], 0)
            TOT_TOKS += data['ctx_len']

        print("Total tokens:          : ", TOT_TOKS)
        print("Total non-pad tokens   : ", NON_PAD_TOKS)
        print("Total pad tokens       : ", PAD_TOKS, f"({(PAD_TOKS / TOT_TOKS) * 100:2.2f}% of total tokens)")
        print(f"Total truncated tokens :  {TRUNCATED_TOKS} ({TRUNCATED_TOKS / TOT_TOKS * 100:2.2f}% of total tokens)")

    if return_list:
        return new_dataset
    return Dataset.from_list(new_dataset)


def json_toolcall_to_python(tool_calls: dict, markdown_format=True) -> str:
    """Convert a JSON tool-call into a Python-style function call string."""

    def format_value(value):
        """Format Python values for code representation."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif value is None:
            return "None"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return "[" + ", ".join(format_value(v) for v in value) + "]"
        elif isinstance(value, dict):
            return (
                "{"
                + ", ".join(
                    f"{format_value(k)}: {format_value(v)}" for k, v in value.items()
                )
                + "}"
            )
        else:
            raise ValueError(f"Unsupported argument type: {type(value)}")

    if isinstance(tool_calls, str):
        tool_calls = json.loads(tool_calls)
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    returns = []

    for tool_call in tool_calls:
        func_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if not func_name:
            raise ValueError("Missing function name in tool call.")

        args_str = ", ".join(
            f"{key}={format_value(val)}" for key, val in arguments.items()
        )
        returns.append(f"{func_name}({args_str})")
    if markdown_format:
        return f"```python\n{'\n'.join(returns)}\n```"
    return returns


def json_tooldef_to_python(tools: list, indent=None) -> str:
    """
    Convert JSON tool descriptions into Python function definitions (as a string).

    - Maps JSON schema types to Python types
    - Handles required vs optional parameters
    - Converts enums to Literal types
    - Adds docstrings from descriptions
    """

    def map_type(prop: dict) -> str:
        """Map JSON schema types to Python type hints."""
        t = prop.get("type", "any")
        if not isinstance(t, str):
            t = 'any'

        # Handle enum -> Literal
        if "enum" in prop or "enums" in prop:
            values = prop.get("enum") or prop.get("enums")
            literals = ", ".join(repr(v) for v in values)
            return f"Literal[{literals}]"

        return {
            "string": "str",
            "str": "str",
            "integer": "int",
            "int": "int",
            "number": "float",
            "float": "float",
            "boolean": "bool",
            "array": "List[Any]",
            "list": "List[Any]",
            "object": "object",
            "any": "Any",
            "null": "None",
        }.get(t, "Any")

    lines = []
    if indent is None:
        indent = random.randint(0, 6)

    if isinstance(tools, str):
        try:
            tools = tool_parse(tools)
        except Exception as E:
            print(tools)
            raise E

    for tool in tools:
        if isinstance(tool, str):
            try:
                tool = tool_parse(tools)
            except Exception as E:
                print(tool)
                raise E
        # print(tool)
        name = tool["name"]
        desc = tool.get("description", "")
        params = tool.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        args_list = []
        doc_params = []

        for p_name, p_info in props.items():
            # print(p_name, '->', p_info)
            py_type = map_type(p_info)
            # Required vs optional
            if p_name in required:
                arg = f"{p_name}: {py_type}"
            else:
                arg = f"{p_name}: Optional[{py_type}] = None"
            args_list.append(arg)

            # Parameter doc
            p_desc = p_info.get("description")
            if p_desc:
                doc_params.append(f"{' '*indent}{p_name}: {p_desc}")

        args_str = ", ".join(args_list)
        
        # Build function string
        func_def = f"def {name}({args_str}):\n"
        func_def += f'{" "*indent}"""{desc}\n\n'
        if doc_params:
            func_def += f"{' '*indent}Args:\n" + "\n".join(doc_params) + "\n"
        func_def += f'{" "*indent}"""\n'
        func_def += f"{' '*(indent)}pass\n"

        lines.append(func_def)

    return "\n\n".join(lines)


if __name__ == "__main__":
    tools = [
        {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
        {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    },
                    "additional_inputs": {"type": "string", "enums": ["YES", "NO"]},
                },
                "required": ["transaction_id"],
            },
        },
    ]

    json_call = json.dumps(
        [
            {
                "name": "web_search",
                "arguments": {
                    "search_str": "Dr Yunus",
                    "result": 5,
                    "paginate": False,
                    "kwargs": {"engine": "google", "grab_index": [1, 3, None]},
                },
            }
        ]
    )
    python_call = 'web_search(search_str="Dr Yunus",)'  # paginate=False, result=5, kwargs={\"search_engine\": \"google\"}, grab=[1, 3, \"None\"])"
    print(json_toolcall_to_python(tools))
    print(json_tooldef_to_python(json_call))
    