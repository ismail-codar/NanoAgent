import json
import re
from ast import literal_eval
from copy import deepcopy
from typing import List

import torch
# import wikipedia
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
)

from typing import Any, Dict, List, Optional, Union


def _tool_parse(tool_call: str) -> Optional[Union[Dict, List[Dict]]]:
    """Parse tool call string using literal_eval and json.loads."""
    try:
        return literal_eval(tool_call)
    except:
        pass
    try:
        return json.loads(tool_call)
    except:
        pass
    return None


def parse_tool_calls(response: str) -> Optional[List[Dict]]:
    """Parse tool calls from model response using regex extraction and JSON parsing."""
    if not response:
        return None

    # Try to extract from ```json ... ``` blocks
    pattern = re.compile(r"```json\s(.*?)```", re.DOTALL)
    tool_calls = pattern.findall(response)

    # Fallback for old approach
    if not tool_calls:
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        tool_calls = pattern.findall(response)

    if tool_calls:
        parsed = _tool_parse(tool_calls[0])
        if parsed is not None:
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                # Single tool call - wrap in list
                return [parsed]

    return None


def search_tool(search_str, results=2):
    import wikipedia
    search_titles = wikipedia.search(search_str, results=results)
    ret_str = ""

    for title in search_titles:
        summary = wikipedia.summary(title, auto_suggest=False)
        if ret_str != "":
            ret_str += "\n\n"
        ret_str += f"# {title}\n\n{summary}"

    return ret_str
