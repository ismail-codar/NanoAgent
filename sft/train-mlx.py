# https://github.com/ml-explore/mlx-examples/blob/main/transformer_lm/main.py
# https://ml-explore.github.io/mlx/build/html/install.html
# MNIST example: https://github.com/ml-explore/mlx-examples/blob/main/mnist/main.py
# Linear example: https://ml-explore.github.io/mlx/build/html/examples/linear_regression.html

import json
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'

import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import generate, load, convert
from mlx_lm.utils import load_model, save_model, dequantize_model
from tqdm import tqdm
from utils.tokenizer import get_tokenizer
from utils.utils import grad_checkpoint


@dataclass
class TrainConfig:
    # Base model config
    MODEL = "quwsarohi/NanoAgent-135M"
    # Iterations
    EPOCHS = 2.1
    BATCH_SIZE = 1
    CONTEXT_LEN = 1024 * 2
    LOAD_PREV = False
    # Learning rate
    MIN_LEARNING_RATE = 0  # 5e-8
    WARMUP_STEPS = int(0.1 * 15000) # 101453
    # SQRT Scaling rule: lr_new = lr * batch_scale = 3e-3 * sqrt(1/128) = ~2.5e-04
    # Ref:
    # * On the SDEs and Scaling Rules for Adaptive Gradient Algorithms
    # * https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Full SFT LR: 1e-4
    # Continued SFT LR: 1e-5
    MAX_LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.1
    QUANTIZATION = None
    GRADIENT_CHECKPOINT_LAYERS = None
    SAVE_PATH = f"weights/{MODEL.split('/')[-1]}-v13"


config_dict = {
    k: v
    for k, v in TrainConfig.__dict__.items()
    if not k.startswith("__") and not callable(v)
}
print(json.dumps(config_dict, indent=2))

cache_mlx_path = os.path.join('weights', TrainConfig.MODEL.split('/')[-1])
if not os.path.exists(cache_mlx_path):
    print('Downloading model and creating mlx model cache at:', cache_mlx_path)
    convert(TrainConfig.MODEL, mlx_path=cache_mlx_path, q_bits=TrainConfig.QUANTIZATION)
    print("Restart training")
    sys.exit()

print("Model loading from:", cache_mlx_path)
model, tokenizer = load(cache_mlx_path)
# model.generation_config.pad_token_id = tokenizer.pad_token_id
# model.generation_config.eos_token_id = tokenizer.eos_token_id

if TrainConfig.QUANTIZATION is not None:
    print("WARNING: QUANTIZATION WOULD MAKE SOME/MOST PARAMETERS UNTRAINABLE")
    nn.quantize(model, group_size=64, bits=TrainConfig.QUANTIZATION)
    print(f"Model quantized to {TrainConfig.QUANTIZATION} bits")
model.train()
tokenizer = get_tokenizer(TrainConfig.MODEL, add_bos=False)


# Gradient checkpointing: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py
if TrainConfig.GRADIENT_CHECKPOINT_LAYERS is not None:
    tot_layers = len(model.layers)
    nlayers = min(tot_layers, TrainConfig.GRADIENT_CHECKPOINT_LAYERS)
    for layer in model.layers[:nlayers]:
        grad_checkpoint(layer)


print(
    f"Memory consumption: {mx.get_active_memory() / 1024 / 1024:.2f} | {mx.get_peak_memory() / 1024 / 1024:.2f} Megabytes"
)


class Dataset:
    """
    Memory optimized dataset. Only converts into tokens when necessary.
    Context stride helps the LLM to go through the sequence
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        shuffle=True,
        stride=None,
        assistant_prefix=None,
        assistant_end=None,
        plw=0,  # prompt loss weight
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cache = None
        self.stride = stride
        self.cache_idx = -1
        self.cache_len = 0
        self.shuffle = shuffle
        if self.stride is not None:
            self._get_len()
        else:
            self.indices = [(i, 0) for i in range(len(self.dataset))]
        # Instruction Tuning Loss over Instructions
        if assistant_prefix:
            assert isinstance(assistant_end, str) and assistant_end is not None
            self.assistant_prefix = self.tokenizer(assistant_prefix)["input_ids"]
            self.assistant_end = self.tokenizer(assistant_end)["input_ids"]
            assert len(self.assistant_end) == 1
            self.assistant_end = self.assistant_end[0]
            self.plw = plw
            print("Prefix start:", self.assistant_prefix)
            print("Prefix end:", self.assistant_end)
        else:
            self.assistant_prefix = None
            self.assistant_end = None

    def _get_len(self):
        self.indices = []
        print("Computing dataset length")
        for idx in tqdm(range(len(self.dataset))):
            self.gen(idx)
            for i in range(self.cache_len):
                self.indices.append((idx, i))
        print("Total length of data:", len(self.indices))

    def __len__(self):
        return len(self.indices)

    def gen(self, idx):
        n_seq = list(range(len(self.dataset[idx]["messages"])))
        messages = []
        if self.shuffle:
            random.shuffle(n_seq)
        for i in n_seq:
            messages += self.dataset[idx]["messages"][i]
        self.cache = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        ctx_len_data = TrainConfig.CONTEXT_LEN  # self.dataset[idx]["train_ctx"]
        self.cache = self.cache.strip()
        if self.stride is None:
            self.cache = self.tokenizer(
                self.cache,
                max_length=ctx_len_data,
                truncation=True,
                return_overflowing_tokens=False,  # Return the overflowing tokens
                return_attention_mask=False,
                padding="max_length",
            )
            self.cache["input_ids"] = [self.cache["input_ids"]]
            # self.cache["attention_mask"] = [self.cache["attention_mask"]]
        else:
            self.cache = self.tokenizer(
                self.cache,
                max_length=ctx_len_data,
                truncation=True,
                return_overflowing_tokens=True,  # Return the overflowing tokens
                stride=self.stride,
                padding="max_length",
                return_attention_mask=False,
            )
        self.cache_idx = idx
        self.cache_len = len(self.cache["input_ids"])
        return ctx_len_data

    def past_assistant_toks(self, idx, tokens):
        p = len(self.assistant_prefix) - 1
        q = idx
        if q < p:
            return False
        while p >= 0:
            if self.assistant_prefix[p] != tokens[q]:
                return False
            p, q = p - 1, q - 1
        return True

    def loss_over_instructions(self, tokens, labels, w=0):
        assistant_prompt = False
        for idx in range(len(tokens)):
            if (not assistant_prompt) and self.past_assistant_toks(idx, tokens):
                assistant_prompt = True
            if assistant_prompt and (tokens[idx] == self.assistant_end):
                assistant_prompt = False
            if not assistant_prompt:
                labels[idx] = w
        return labels

    def __getitem__(self, idx):
        p, q = self.indices[idx]
        ctx_len = None
        if self.cache_idx != p:
            ctx_len = self.gen(p)
        x = self.cache["input_ids"][q]
        y = x[1:] + [self.tokenizer.pad_token_id]
        # Cross entropy loss weights
        weights = [0 if t == self.tokenizer.pad_token_id else 1 for t in y]
        if self.assistant_prefix:
            weights = self.loss_over_instructions(x, weights, w=self.plw)
        return {
            "x": mx.array([x]),
            "y": mx.array([y]),
            "weights": mx.array([weights]),
            "ctx": ctx_len,
        }


def source_dist(dataset):
    # Source distribution
    SOURCE = dict()
    for d in dataset:
        for s in d["source"]:
            if s not in SOURCE:
                SOURCE[s] = 0
            SOURCE[s] += 1

    print("Source distribution", flush=True)
    print("--------------------", flush=True)
    pack_len = 0
    for k, v in SOURCE.items():
        print(k, ":", v, flush=True)
        pack_len += v
    print(f"\nTotal data: {len(dataset)} (Unpacked len: {pack_len})")
    print("\n\n")


# Uncomment to test user query masking
# test_dataset = Dataset(
#     dataset=[
#         {
#             "messages": [
#                 [
#                     {"role": "user", "content": "Hi"},
#                     {"role": "assistant", "content": "Hello, How can I help you today?"}
#                 ],
#                 [
#                     {"role": "user", "content": "9+5=?"},
#                     {"role": "assistant", "content": "9+5=14"}
#                 ]
#             ],
#             "train_ctx": 64
#         }
#     ],
#     shuffle=False,
#     tokenizer=tokenizer,
#     assistant_prefix="<|im_start|>assistant\n",
#     assistant_end="<|im_end|>",
#     plw=0.0
# )
# dat = test_dataset[0]
# for i in range(64):
#     print(f"{i}:{tokenizer.decode(dat['x'][0][i].item())}->{tokenizer.decode(dat['y'][0][i].item())}: {dat['weights'][0][i].item()}")

# for t in tokenizer.encode("<|im_start|>assistant\n"):
#     print(t, tokenizer.decode(t))
# sys.exit()

train_ds = load_dataset(
    "json",
    data_files=f"data/datasets/Smollm2_base_train_{TrainConfig.CONTEXT_LEN}_v13.jsonl",
    split="train",
)
test_ds = load_dataset(
    "json",
    data_files=f"data/datasets/Smollm2_base_test_{TrainConfig.CONTEXT_LEN}_v13.jsonl",
    split="train",
)
# dataset = dataset.sort('ctx_len')

print("Train Dataset:\n---------------")
source_dist(train_ds)
dataset = Dataset(
    dataset=train_ds,
    shuffle=True,
    tokenizer=tokenizer,
    assistant_prefix="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    plw=0.0,
)

print("Test Dataset:\n---------------")
source_dist(test_ds)
eval_dataset = Dataset(
    dataset=test_ds,
    shuffle=False,
    tokenizer=tokenizer,
    assistant_prefix="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    plw=0.0,
)


def cosine_decay_with_warmup(
    max_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0,
):
    def schedule(step):
        # Linear warmup
        linear_warmup = max_lr * step / warmup_steps
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + mx.cos(mx.pi * progress))
        cosine_decay = (max_lr - min_lr) * cosine_decay + min_lr
        return mx.where(
            step < warmup_steps,
            linear_warmup,
            cosine_decay
        )

    return schedule


scheduler_adam = cosine_decay_with_warmup(
    max_lr=TrainConfig.MAX_LEARNING_RATE,
    min_lr=TrainConfig.MIN_LEARNING_RATE,
    total_steps=int(len(dataset) * TrainConfig.EPOCHS) // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS,
)

scheduler_muon = cosine_decay_with_warmup(
    max_lr=TrainConfig.MAX_LEARNING_RATE * 10 * 2,
    min_lr=TrainConfig.MIN_LEARNING_RATE,
    total_steps=int(len(dataset) * TrainConfig.EPOCHS) // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS,
)

# optimizer = optim.AdamW(
#     learning_rate=scheduler,
#     weight_decay=TrainConfig.WEIGHT_DECAY,
#     # betas=[0.9, 0.95],
#     # SQRT Scaling: bi = 1 - scale_val * (1 - bi)
#     betas=[0.9, 0.99],
# )

optimizer = optim.MultiOptimizer(
    [        
        optim.Muon(
            learning_rate=scheduler_muon, weight_decay=TrainConfig.WEIGHT_DECAY
        ),
        optim.AdamW(
            learning_rate=scheduler_adam, betas=[0.9, 0.99], weight_decay=TrainConfig.WEIGHT_DECAY, eps=1e-12
        )
    ],
    # Where muon will be applied
    [lambda name, weight: weight.ndim >= 2 and 'embed' not in name and 'norm' not in name]
)


def get_batch(dataset, idx=1, batch_size=1):
    x, y, w, ctx = [], [], [], []
    for _ in range(batch_size):
        data = dataset[idx]
        x.append(data["x"])
        y.append(data["y"])
        w.append(data["weights"])
        ctx.append(data["ctx"])

    return mx.concatenate(x), mx.concatenate(y), mx.concatenate(w), ctx


# x, y, w = get_batch(10)
# x = x.tolist()[0]
# y = y.tolist()[0]
# w = w.tolist()[0]

# inp = ''
# tar = ''
# for i in range(1024):
#     if w[i] == 0:
#         inp += '*'
#     else:
#         inp += tokenizer.decode(x[i])
#     tar += tokenizer.decode(y[i])

# print(inp)
# print('-+'*20)
# print(tar)

def save_state(
    iter_step,
    losses,
    eval_losses,
    model,
    optimizer,
    seen_tokens,
    path=TrainConfig.SAVE_PATH,
):
    if not os.path.exists(path):
        os.makedirs(path)
    # Save optimizer the state
    # https://ml-explore.github.io/mlx/build/html/python/optimizers.html
    mx.eval(model.state, optimizer.state)
    mx.save_safetensors(
        os.path.join(path, "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )
    save_model(save_path=path, model=dequantize_model(deepcopy(model)))

    train_info = {
        "training_params": {
            **config_dict,
            "TRAIN_DATASET_LEN": len(dataset),
        },
        "iter_step": iter_step,
        "losses": losses,
        "eval_losses": eval_losses,
        "seen_tokens": seen_tokens,
    }
    with open(os.path.join(path, "train_info.json"), "w") as f:
        json.dump(train_info, f, indent=2)
    tokenizer.save_pretrained(path)
    # push_trackio(train_info)
    print("\nModel saved", flush=True)


def load_state(path=TrainConfig.SAVE_PATH):
    optimizer.state = tree_unflatten(
        list(mx.load(os.path.join(path, "optimizer.safetensors")).items())
    )
    model, _ = load_model(Path(path))

    with open(os.path.join(path, "train_info.json"), "r") as f:
        train_info = json.load(f)

    mx.eval(model.state, optimizer.state)
    print("Model loaded", flush=True)
    return (
        model,
        optimizer,
        train_info["iter_step"],
        train_info["losses"],
        train_info["eval_losses"],
        train_info["seen_tokens"],
    )


losses, eval_losses = [], {}
itr_start = 0
seen_tokens = 0
log_steps = 1
eval_loss = defaultdict(float)
if TrainConfig.LOAD_PREV:
    del model
    model, optimizer, itr_start, losses, eval_losses, seen_tokens = load_state()
win_loss = sum(losses)
# nn.quantize(model)


# state = [model.state, optimizer.state]
# @partial(mx.compile, inputs=state, outputs=state)
def cal_loss(model, x, y, weights):
    logits = model(x)
    # Shape: (bs, ctx_len)
    tok_loss = nn.losses.cross_entropy(logits, y, reduction="none")
    total_toks = mx.maximum(weights.sum(), 1e-12)
    ce_loss = (tok_loss * weights).sum() / total_toks
    return ce_loss


state = [model.state, optimizer.state]
@partial(mx.compile, inputs=state, outputs=state)
def train_step(x, y, w):
    model.train()
    loss_and_grad_fn = nn.value_and_grad(model, cal_loss)
    loss, grads = loss_and_grad_fn(model, x, y, w)
    clipped_grads, total_norm = optim.clip_grad_norm(grads, max_norm=1.0)
    optimizer.update(model, clipped_grads)
    return loss


# @partial(mx.compile)
def model_eval(tqdm_disable=False):
    eval_loss = []
    model.eval()
    for eval_itr in tqdm(range(len(eval_dataset)), leave=False, disable=tqdm_disable):
        x, y, w, ctx = get_batch(eval_dataset, idx=eval_itr, batch_size=TrainConfig.BATCH_SIZE)
        # toks = w.sum().item()
        eval_loss.append(cal_loss(model, x, y, w).item())
    return sum(eval_loss) / len(eval_loss)


tqdm_data = tqdm(
    range(itr_start, int(len(dataset) * TrainConfig.EPOCHS), TrainConfig.BATCH_SIZE),
    total=int(len(dataset) * TrainConfig.EPOCHS) // TrainConfig.BATCH_SIZE,
    initial=itr_start,
    leave=False,
    # mininterval=2.5
)

eval_loss = model_eval(False)
if itr_start == 0:
    eval_losses[1] = {
        "train_loss": 0.0,
        "eval_loss": eval_loss,
        "lr": optimizer.learning_rate.item(),
    }

mx.eval(state)
total_loss = sum(losses)
for itr in tqdm_data:
    x, y, w, ctx = get_batch(
        dataset, idx=itr % len(dataset), batch_size=TrainConfig.BATCH_SIZE
    )
    toks = w.sum().item()
    loss = train_step(x, y, w).item()
    non_pad_toks = mx.where(x != tokenizer.pad_token_id, 1, 0).sum().item()
    mx.eval(state)
    losses.append(loss)
    total_loss += loss
    seen_tokens += toks

    if (itr + 1) % log_steps == 0:
        tqdm_data.set_description(
            f"TL {total_loss / (itr + 1):1.4f}/ EL {eval_loss:1.4f} / CTX {int(toks), int(non_pad_toks), ctx[0]} | LR {optimizer.learning_rate.item():1.6f}"
        )
        if (itr + 1) % 500 == 0:
            eval_loss = model_eval(False)
            eval_losses[itr + 1] = {
                "train_loss": win_loss / seen_tokens,
                "eval_loss": eval_loss,
                "lr": optimizer.learning_rate.item(),
            }
            save_state(
                iter_step=itr + 1,
                losses=losses,
                eval_losses=eval_losses,
                model=model,
                optimizer=optimizer,
                seen_tokens=seen_tokens,
            )
            print(
                f"Training progress: {((itr + 1) / (len(dataset) * TrainConfig.EPOCHS)) * 100:2.2f}",
                flush=True,
            )

save_state(
    iter_step=int(len(dataset) * TrainConfig.EPOCHS),
    losses=losses,
    eval_losses=eval_losses,
    model=model,
    optimizer=optimizer,
    seen_tokens=seen_tokens,
    path=os.path.join(TrainConfig.SAVE_PATH, 'final_checkpoint')
)