# 🧠 NanoAgent — A 135M Parameter Agentic SLM

NanoAgent is a **135M parameter**, **8k context length**, open-source language model designed for **agentic tasks** such as **tool calling**, **instruction following**, and **lightweight reasoning**.  
It’s small enough (~135 MB in 8-bit) to run on **edge devices** like personal laptops, low-memory CPUs, and even wearables — yet smart enough to make tool calls, parse web information, and give structured answers.

Quick inference resource: [here](notebooks/inference.ipynb)

Huggingface Model: [NanoAgent-135M](https://huggingface.co/quwsarohi/NanoAgent-135M)

Run in Ollama: `ollama run quwsarohi/NanoAgent`

## 🌍 Real-World Use Cases

- 🕹️ **Runs on edge devices** — laptops, smartwatches, browsers, or CPU-only environments.  
- 🌐 **Parses and answers from the web** — supports tool calling to fetch real-time information.  
- 🔎 **Answers recent questions** with live web search tools.  
- 💬 **Continues conversations** — ideal for assistant or agent frameworks.  
- ⚙️ **Tool calling support** enables chaining multiple tools and parsing results to produce final answers.


## ✨ What NanoAgent Supports

| Capability                        | Description                                                                                     | 
|------------------------------------|--------------------------------------------------------------------------------------------------|
| 💬 Basic conversation              | Casual small talk                                                                     |
| 🌐 Information retrieval           | e.g., *“How to bake a cake?”*, *“Weather in Toronto”* through web search. Extracts answers from information returned by tools (scraping/search)                        |
| 🧰 Tool calling                    | Single & multi-tool call with structured explanation                                            |
| 🧠 Question decomposition          | Breaks complex questions into steps                                                             | 
| 🧭 Question classification         | Identifies type of user query (e.g., fact, reasoning, instruction)                              |
| 📝 Following system prompts       | Responds properly to system-level instructions                                                  | 
| ✍️ Writing emails and tasks       | Writes emails, structured messages                                                              | 
---

## 🧪 Training Overview

- **Base model**: [`SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (instruction-tuned)
- **Fine-tuning method**: ~~[Dynamic Fine-Tuning (DFT)](https://github.com/yongliang-wu/DFT/tree/master)~~ Supervised Fine-Tuning
- **Platform**: Apple Mac M1 (16 GB) — MLX framework

### 📚 Datasets Used

This model was trained using a combination of datasets under different open licenses.  
Each dataset retains its original license, and use of those datasets is subject to their respective terms.

#### General Training (SFT)
| Dataset | Purpose | License |
|---------|---------|---------|
| [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | Math reasoning, word-level reasoning | MIT |
| [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) | Instruction following with personas | Open Data Commons License Attribution |
| [mlabonne/orca-agentinstruct-1M-v1-cleaned](https://huggingface.co/datasets/mlabonne/orca-agentinstruct-1M-v1-cleaned) | RAG, MCQ, JSON parsing, text classification | Community Data License Agreement – Permissive, Version 2.0 |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (systemchats-30k) | General conversation, system prompts | Apache-2.0 |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (everyday-conversations) | Everyday conversation | Apache-2.0 |
| [nvidia/Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) | Instruction following, structured outputs | NVIDIA Open Model License |

#### Function Calling Training
| Dataset | Purpose | License |
|---------|---------|---------|
| [Locutusque/function-calling-chatml](https://huggingface.co/datasets/Locutusque/function-calling-chatml) | Tool call response formatting | Apache-2.0 |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Function calling coverage | Creative Commons Attribution 4.0 |
| [nemotron/interactive_agent](https://huggingface.co/datasets/nemotron/interactive_agent) (local) | Tool calling, agentic behavior | NVIDIA Open Model License |


## 🧭 Key Explorations & Findings

- ✂️ **Dataset deduplication** significantly improved performance by removing noisy or duplicate Q/As.  
 - ✂️ **Shortening the responses** (casual response) and using shorter python code in training improved performance and reduce repeated token generation.
- 🧮 **Word-level reasoning** from `orca-math` enhanced the model’s ability to handle stepwise logic.  
- 🧰 Designing tool calling prompts using **six open-source tool calling datasets** resulted in stronger structured output generation.  
- 🌐 Tool calling integration enabled the model to **extract answers from parsed web data**, supporting up-to-date queries.  


## ⚡ Benchmark

### Model Comparison

| Benchmark | SmolLM2-135M-Instruct | NanoAgent |
|-----------|:---------------------:|:---------:|
| **Commonsense QA** (acc) | 20.88% | 20.23% |
| **IFEval** (prompt strict) | 21.63% | **29.94%** |
| **IFEval** (inst strict) | 35.01% | **42.33%** |
| **IFEval** (prompt loose) | 23.84% | **32.16%** |
| **IFEval** (inst loose) | 37.65% | **45.32%** |
| **tinyArc** (acc_norm) | 33.76% | 36.47% |
| **tinyGSM8k** (exact_match) | 0.55% | 2.31% |
| **tinyHellaswag** (acc_norm) | 42.20% | **43.45%** |
| **tinyMMLU** (acc_norm) | 26.79% | **27.62%** |
| **tinyTruthfulQA** (acc) | 38.65% | **40.45%** |
| **tinyWinogrande** (acc_norm) | 46.48% | 42.86% |

### BFCL Benchmark (Tool Calling)

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| **Overall** | 24.35% | 609/2501 |
| parallel | 50.50% | 101/200 |
| parallel_multiple | 48.00% | 96/200 |
| simple_python | 33.75% | 135/400 |
| simple_javascript | 32.00% | 16/50 |
| multiple | 25.00% | 50/200 |
| live_simple | 24.03% | 62/258 |
| simple_java | 22.00% | 22/100 |
| live_parallel | 18.75% | 3/16 |
| live_parallel_multiple | 16.67% | 4/24 |
| live_multiple | 11.40% | 120/1053 |

### Key Findings

- **NanoAgent** significantly outperforms the base **SmolLM2-135M-Instruct** on **instruction following** (IFEval) with +8-10% improvements across all metrics
- **NanoAgent** improves on **tinyMMLU**, **tinyTruthfulQA**, and **tinyHellaswag** over the base model
- 🧰 **Tool Calling**: Only NanoAgent supports tool calling — SmolLM2-135M-Instruct does not


## ⚡ Example Usage

### Basic Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "quwsarohi/NanoAgent-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def inference(messages, max_new_tokens=256, temperature=0.3, **kwargs):
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        **kwargs
    )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

messages = [{"role": "user", "content": "Hi! Do you have a name?"}]
print(inference(messages))
```

### Tool Calling
NanoAgent uses a JSON-based tool calling format:

````python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a web search and returns formatted results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        }
    }
]

TOOL_TEMPLATE = """You are a helpful AI assistant. You have a set of possible tools that you can execute to retrieve information or to perform specific actions. You can execute zero or more tools to answer user question.

Here are the list of tools that you have access to:
```json
{tools}
```

Only execute tools from above. Follow the below JSON signature to execute tools:
```json
[{{"name": "tool_name", "arguments": {{"arg1": "val1", ...}}}}, ...]
```
"""

messages = [
    {"role": "system", "content": TOOL_TEMPLATE.format(tools=json.dumps(tools, indent=2))},
    {"role": "user", "content": "What's the latest AI news?"},
]
response = inference(messages, max_new_tokens=512)
print(response)

# Output: ```json
# [{"name": "web_search", "arguments": {"query": "latest AI news 2026"}}]
# ```
````


## 🧭 Roadmap

- [ ] 📊 Benchmark more agentic tasks  
- [ ] 🧠 Explore GRPO for tool calling improvement  
- [ ] 🔀 Experiment with weight merging  
- [ ] 🧪 Evaluate multi-turn tool chaining  
- [ ] 🧹 Further refine datasets for stability


## Directory Tree

```
NanoAgent/
├── data/
│   ├── dataprep.py          # Dataset preparation, cleaning, and formatting
│   └── utils.py             # Helper utilities for data processing
│
├── grpo/
│   └── grpo-mlx.py          # Experimental GRPO (agentic fine-tuning) implementation using MLX
│
├── notebooks/
│   └── inference.ipynb      # Demo notebook for inference and evaluation
│
├── sft/
│   └── train-mlx.py         # Supervised Fine-Tuning (SFT) training script using MLX
│
├── utils/
│   ├── gguf_conv.py         # Conversion script for exporting model to GGUF format (for llama.cpp etc.)
│   ├── tokenizer.py         # Tokenizer helper functions and configs
│   └── webtool.py           # Example tool interface for web search / parsing integration
│
├── LICENSE                  # Apache 2.0 license file
├── NOTICE                   # Notices and attributions for datasets and dependencies
└── README.md                # Project overview, usage guide, and dataset details
```

---

## 📄 License

This project (code, model weights, and training recipes) is licensed under the [Apache License 2.0](./LICENSE).

## 📢 Notice

- Model & code are © [quwsarohi](https://github.com/QuwsarOhi), licensed under Apache 2.0.  
- Portions of the training data were sourced from third-party datasets under CDLA-P 2.0, MIT, CC-BY 4.0, ODC-BY, and Apache 2.0.  
- The licensors of these datasets do **not endorse** this project or its outputs.  
- If you redistribute or fine-tune this model, ensure your use complies with all applicable dataset licenses.


