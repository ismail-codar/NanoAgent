# References: 
# https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb
# https://abderrahmanskiredjgithub.io/the-illustrated-grpo/The%20Illustrated%20GRPO.pdf
# https://github.com/huggingface/trl/issues/3662
# https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig
# https://towardsdatascience.com/how-to-finetune-small-language-models-to-think-with-reinforcement-learning/

import json
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import warnings
warnings.filterwarnings("ignore")

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from collections import defaultdict
import pickle
from functools import partial
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import tqdm
import random
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from mlx_lm import batch_generate, generate, stream_generate, load, convert
from mlx_lm.utils import load_model, save_model, dequantize_model
from mlx_lm.sample_utils import make_sampler, make_logits_processors
# from data.grpo.salseforce_tool import salesfores_toolcall

from data.grpo.websearch_tool import tool_calling_traces
from data.grpo.mobile_actions import mobileactions
from data.grpo.autoif import autoif_ds
from data.grpo.ifeval import ifeval_ds
# from data.grpo.gorilla_tool import gorilla_openfun
from data.grpo.reasoning_gym import *
from data.grpo.general_chat import general_chat_ds

from scipy.ndimage import gaussian_filter1d
from utils.utils import grad_checkpoint, split_grads

os.environ["TOKENIZERS_PARALLELISM"] = "true"
plt.ioff()
matplotlib.use("Agg")

from dataclasses import dataclass
from utils.tokenizer import get_tokenizer
from utils.webtool import tool_call_extract


@dataclass
class TrainConfig:
    # Iterations
    ITERS = 3_000
    GENERATE_DATA = True
    BATCH_SIZE = 1
    GEN_LEN = 256 + 128
    SAVE_FREQ = 50
    LOAD_PREV = False
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0
    EPSILON_MIN = 3e-3    # Sequence/GSPO: 3e-4 | GRPO: 0.2 |   Note: Should not be changed
    EPSILON_HIGH = 4e-3   # Sequence/GSPO: 4e-4 | GRPO: 0.272 | Note: Can be changed 
    GROUP_SIZE = 8
    WARMUP_STEPS = 50 # 50
    DECAY_STEPS = 40 # 10
    BETA = 0.04 # 0.04
    EVAL_STEPS = 50
    NUM_MODEL_UPDATE_MU = 1   # Number of backpropagations per question/rollout
    GRAD_ACCUM = 1
    GRAD_NORM = 1
    REF_MODEL_MIXUP_ALPHA = 0 # 0.6
    MAX_INPUT_LEN = 256 # 768
    SAVE_PATH = "weights/NanoAgent-135M-grpo-ifeval"
    DATA_PATH = "data/datasets/grpo_cache.pickle"
    MODEL = "quwsarohi/NanoAgent-135M-think" #"weights/SmolLM2-360M-mlx-Instruct" #"quwsarohi/NanoAgent-135M"
    FREEZE_LAYERS = [] # embed_tokens
    QUANTIZATION = None
    GRADIENT_CHECKPOINT_LAYERS = 6
    EVAL_SAMPLES = 100
    TQDM = True
    STD_NORM = False
    CONST_TOK_SCALE = False
    SAMPLING = 'custom' #'sequence'
    SOFT_CLIP = False # Soft clipping proposed in SAPO paper - https://arxiv.org/pdf/2511.20347
    TEMPERATURE = 0.8 # Better to keep <= 0.9
    MIN_P = None # Expected ~0.2 for Smollm2-135M
    TOP_K = None
    TOP_P = 0.9
    REPETITION_PENALTY = 1.1 # ~1.1

# GSPO Constraints:
# -----------------
# STD_NORM = True
# CONST_TOK_SCALE = False
# SOFT_CLIP = False
# EPSILON_MIN = 3e-3  (minimum value that updates logprob) # Sequence/GSPO: 3e-4
# EPSILON_HIGH = 4e-3 (minimum value that updates logprob) # Sequence/GSPO: 4e-4
# SAMPLING = "sequence"

# DR GRPO Constraints:
# --------------------
# STD_NORM = False
# CONST_TOK_SCALE = True
# SOFT_CLIP = False
# EPSILON_MIN = 0.2
# EPSILON_HIGH = 0.272
# SAMPLING = "token"

# SAPO Constraints:
# -----------------
# STD_NORM = False
# CONST_TOK_SCALE = True
# SOFT_CLIP = True [main constraint]
# EPSILON_MIN = 0.2 [unused]
# EPSILON_HIGH = 0.272 [unused]
# SAMPLING = "token"
# t_pos=1, t_neg=1.05

# Token Sampling: DAPO - https://arxiv.org/pdf/2503.14476
# Sequence Sampling: GSPO - https://arxiv.org/pdf/2507.18071

assert TrainConfig.SAMPLING in ['token', 'sequence', 'custom']
assert 0 < TrainConfig.GRAD_NORM or TrainConfig.GRAD_NORM is not None
assert 1 <= TrainConfig.GRAD_ACCUM
assert 1 == TrainConfig.BATCH_SIZE
assert 0 <= TrainConfig.WEIGHT_DECAY

if TrainConfig.QUANTIZATION is not None:
    print("WARNING: QUANTIZATION WOULD MAKE SOME/MOST PARAMETERS UNTRAINABLE")

config_dict = {
    k: v
    for k, v in TrainConfig.__dict__.items()
    if not k.startswith("__") and not callable(v)
}
print(json.dumps(config_dict, indent=2))

# The model that will be trained

cache_mlx_path = os.path.join('weights', TrainConfig.MODEL.split('/')[-1])
if not os.path.exists(cache_mlx_path):
    print('Downloading model and creating mlx model cache at:', cache_mlx_path)
    convert(TrainConfig.MODEL, mlx_path=cache_mlx_path, q_bits=TrainConfig.QUANTIZATION)
    print("Restart training")
    sys.exit()


model, tokenizer_mlx, model_config = load(cache_mlx_path, return_config=True)

if TrainConfig.QUANTIZATION is not None:
    nn.quantize(model, group_size=64, bits=TrainConfig.QUANTIZATION)
    print(f"Model quantized to {TrainConfig.QUANTIZATION} bits")

if TrainConfig.BETA > 0:
    # The reference model for KL-div (freezed)
    model_ref = load(cache_mlx_path, return_config=False)[0] #deepcopy(model)
    model_ref.eval().freeze()
else:
    model_ref = None

tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-135M", add_bos=False)
# tokenizer = get_tokenizer("google/functiongemma-270m-it", add_bos=True)
assert tokenizer.pad_token_id != tokenizer.eos_token_id, "Pad token and EOS token must be different"

# Learning Rate Schedulers
def linear_decay_with_warmup(
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
):
    assert total_steps > warmup_steps + decay_steps
    def schedule(step):
        # Linear warmup: 0 → base_lr
        warmup_lr = base_lr * step / warmup_steps
        # Linear decay: base_lr → 0
        decay_progress = (step - (total_steps - decay_steps)) / decay_steps
        decay_lr = base_lr * (1.0 - decay_progress)
        return mx.where(
            step < warmup_steps,
            warmup_lr,
            mx.where(
                step >= (total_steps - decay_steps),
                mx.maximum(decay_lr, 0.0),
                base_lr
            )
        )
    return schedule


scheduler = linear_decay_with_warmup(
    base_lr=TrainConfig.LEARNING_RATE,
    total_steps=(TrainConfig.ITERS * TrainConfig.NUM_MODEL_UPDATE_MU) // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS * TrainConfig.NUM_MODEL_UPDATE_MU,
    decay_steps=TrainConfig.DECAY_STEPS * TrainConfig.NUM_MODEL_UPDATE_MU
)

scheduler_muon = linear_decay_with_warmup(
    base_lr=TrainConfig.LEARNING_RATE * 10 * 2, #0.001, 0.01, 0.02, 1e-3,
    total_steps=TrainConfig.ITERS // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS,
    decay_steps=TrainConfig.DECAY_STEPS
)

# optimizer = optim.AdamW(
#     learning_rate=scheduler, betas=[0.9, 0.999], weight_decay=TrainConfig.WEIGHT_DECAY, eps=1e-12
# )

# Interesting writings:
# * https://huggingface.co/blog/onekq/muon-optimizer
# * https://www.lakernewhouse.com/writing/muon-2
# * https://kellerjordan.github.io/posts/muon/
# * https://varunneal.github.io/essays/muon
# * https://github.com/KellerJordan/Muon

optimizer = optim.MultiOptimizer(
    [        
        optim.Muon(
            learning_rate=scheduler_muon, weight_decay=TrainConfig.WEIGHT_DECAY
        ),
        optim.AdamW(
            learning_rate=scheduler, betas=[0.9, 0.999], weight_decay=TrainConfig.WEIGHT_DECAY, eps=1e-12
        )
    ],
    # Where muon will be applied
    [lambda name, weight: weight.ndim >= 2 and 'embed' not in name and 'norm' not in name]
)

def total_tokens(data):
    return len(
        tokenizer.encode(
            data,
        )
    )


def tool_tokens(ground_tool_call):
    ntokens = len(tokenizer.encode(json.dumps(ground_tool_call)))
    return ntokens


if TrainConfig.GENERATE_DATA:
    ds_size = 4 * TrainConfig.ITERS
    train_ds = []

    # --- IF Eval ---
    # sz = int(ds_size * 0.4)
    # train_ds += autoif_ds(tokenizer, TrainConfig.MAX_INPUT_LEN, n_instructions=1)
    # train_ds = sorted(train_ds, key=lambda x: len(x['prompt']), reverse=True)#[:sz]
    sz = int(ds_size * 0.6)
    train_ds += ifeval_ds(tokenizer, TrainConfig.MAX_INPUT_LEN, n_instructions=1, kshot=False)#[:sz]
    # sz = int(ds_size * 0.4)
    train_ds += sorted(general_chat_ds(tokenizer, TrainConfig.MAX_INPUT_LEN), key=lambda x: len(x['prompt']), reverse=True)[:sz]
    # random.shuffle(train_ds)
    # train_ds = train_ds[:sz]
    # print("IF-Eval DS size:", len(train_ds))

    
    # --- Tool Call ---
    # sz = int(ds_size * 0.5)
    # train_ds += tool_calling_traces(tokenizer, TrainConfig.MAX_INPUT_LEN)[:sz]
    # sz = int(ds_size * 0.025)
    # train_ds += mobileactions(tokenizer, TrainConfig.MAX_INPUT_LEN)[:sz]

    # --- Math Mix ---
    # sz = int(ds_size * 0.25)
    # train_ds += alice_in_wonderland(tokenizer=tokenizer, size=sz, think=False)
    # sz = int(ds_size * 0.1)
    # train_ds += syllogism(tokenizer, size=sz, think=False)
    # sz = int(ds_size * 0.3)
    # train_ds += gsm_symbolic(tokenizer, size=sz, think=False)
    # sz = int(ds_size * 0.15)
    # train_ds += chain_sum(tokenizer, size=sz, think=False)
    # sz = int(ds_size * 0.1)
    # train_ds += zebra_puzzles(tokenizer, size=sz, think=False)
    # sz = int(ds_size * 0.1)
    # train_ds += needle_haystack(tokenizer, size=sz*3, prompt_token_len=TrainConfig.MAX_INPUT_LEN, think=False)[:sz]
    
    random.shuffle(train_ds)
    print("New Generated Dataset length:", len(train_ds))
    with open(TrainConfig.DATA_PATH, 'wb') as fp:
        pickle.dump(train_ds, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Cache file created on", TrainConfig.DATA_PATH)
else:
    with open(TrainConfig.DATA_PATH, 'rb') as fp:
        train_ds = pickle.load(fp)
    print(
        f"Dataset loaded from path: {TrainConfig.DATA_PATH} | Dataset length: {len(train_ds)}"
    )
if TrainConfig.EVAL_SAMPLES:
    eval_ds = train_ds[-TrainConfig.EVAL_SAMPLES:]
    train_ds = train_ds[:-TrainConfig.EVAL_SAMPLES]
else:
    eval_ds = None


def evaluate(eval_model, runs=4, temp=0.):
    # return [0]
    if not eval_ds:
        return [0]
    rewards = []
    eval_model.eval()
    for idx in tqdm.tqdm(range(len(eval_ds)), leave=False):
        # Removing tool_call lead
        data = eval_ds[idx]
        prompt_tokens = tokenizer.encode(data['prompt'])
        scorer = data['scorer']
        for _ in range(runs):
            response = generate(
                eval_model,
                tokenizer,
                prompt_tokens,
                max_tokens=TrainConfig.GEN_LEN,
                sampler=None if temp == 0 else lambda x: mx.random.categorical(x / temp, axis=-1)
            )
            rewards.append(scorer(response, False))
    return rewards


def mean_map(data, win=20):
    def _mean(x):
        # while len(x) < win: x.append(min(x))
        return sum(x) / len(x)
    _data = []
    for i in range(len(data)):
        data_win = data[max(0, i-win+1):i+1]
        _data.append(_mean(data_win))
    return _data


def prog_graph(
    iter,
    all_losses,
    learning_rates,
    all_rewards,
    eval_rewards,
    std_rewards,
    save_path=None,
    plot=True,
):
    plt.close()
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), dpi=600)
    fig.suptitle(f"GRPO Iter: {iter}", fontsize=13)

    # GRPO Loss
    axes[0].plot(np.cumsum(all_losses) / (np.arange(len(all_losses)) + 1), color="tab:red", alpha=0.6, linestyle=':', label='Cumulative Sum')
    # axes[0].plot(mean_map(all_losses), color="tab:red", alpha=0.6, linestyle='--', label='Mean Win. 20')
    axes[0].plot(gaussian_filter1d(all_losses, sigma=2), color="tab:red", linewidth=2, label='Smoothen')
    # axes[0].plot(all_losses, color="tab:red", alpha=0.2)
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Rewards
    all_rewards = np.asarray(all_rewards)
    std_rewards = np.asarray(std_rewards)
    axes[1].plot(np.cumsum(all_rewards) / (np.arange(len(all_rewards)) + 1), color="tab:blue", alpha=0.8, linestyle=':', label='Cumulative Sum')
    axes[1].plot(mean_map(all_rewards), color="tab:blue", alpha=0.8, linestyle='--', label='Mean Win. 20')
    axes[1].plot(gaussian_filter1d(all_rewards, sigma=2.5), linewidth=2, color="tab:blue", label='Smoothen')
    # axes[1].fill_between(
    #     np.arange(len(all_rewards)),
    #     all_rewards - std_rewards,
    #     all_rewards + std_rewards,
    #     color="tab:blue",
    #     alpha=0.15,
    #     # label="±1 Std"
    # )
    # axes[1].plot(all_rewards, alpha=1, color="tab:blue", label='Reward (batch-mean)')
    axes[1].set_title("Rewards")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(bottom=0)

    itrs = [x['iter'] for x in eval_rewards]
    eval_greedy = [e['eval_score'] for e in eval_rewards]

    axes[2].scatter(itrs, eval_greedy, color="tab:green", linewidth=2, marker="*")
    axes[2].plot(itrs, eval_greedy, color="tab:green", linewidth=2)

    # axes[2].scatter(itrs, eval_sampling, color="tab:green", linewidth=2, marker="x")
    # axes[2].plot(itrs, eval_sampling, color="tab:green", alpha=0.6, linestyle='--', linewidth=2)

    axes[2].set_title("Eval Rewards")
    axes[2].grid(True)

    # total_prompt_tokens
    axes[3].plot(learning_rates, color="tab:orange")
    axes[3].set_title("Learning Rate")
    axes[3].grid(True)

    # Remove [0, 1] default ticks on x-axis for bottom graph
    plt.tight_layout(pad=1)
    if save_path:
        plt.savefig(
            f"{save_path}/plot.jpg", dpi=400, bbox_inches="tight", transparent=True
        )
    if plot:
        plt.show()


def load_state(path=TrainConfig.SAVE_PATH):
    optimizer.state = tree_unflatten(
        list(mx.load(os.path.join(path, "optimizer.safetensors")).items())
    )
    model = load(Path(path), model_config=model_config)[0]

    with open(os.path.join(path, "train_info.json"), "r") as f:
        train_info = json.load(f)

    mx.eval(model.state, optimizer.state)
    print("Model loaded", flush=True)
    return (
        model,
        optimizer,
        train_info["iter_step"],
        train_info["losses"],
        train_info["learning_rates"],
        train_info["all_rewards"],
        train_info["max_rewards"],
        train_info["std_rewards"],
        train_info["eval_rewards"]
    )


if TrainConfig.LOAD_PREV:
    del model
    (
        model,
        optimizer,
        iter_step,
        losses,
        learning_rates,
        all_rewards,
        max_rewards,
        std_rewards,
        eval_rewards
    ) = load_state(TrainConfig.SAVE_PATH)
    print("Previous weights loaded")
else:
    (
        iter_step,
        losses,
        learning_rates,
        all_rewards,
        max_rewards,
        std_rewards,
        eval_rewards
    ) = 0, [], [], [], [], [], []


if TrainConfig.GRADIENT_CHECKPOINT_LAYERS is not None:
    tot_layers = len(model.layers)
    nlayers = min(tot_layers, TrainConfig.GRADIENT_CHECKPOINT_LAYERS)
    for layer in model.layers[:nlayers]:
        grad_checkpoint(layer)

print(
    f"Memory consumption: {mx.get_active_memory() / 1024 / 1024:.2f} | {mx.get_peak_memory() / 1024 / 1024:.2f} Megabytes"
)

def save_state(
    iter_step,
    losses,
    learning_rates,
    all_rewards,
    max_rewards,
    std_rewards,
    eval_rewards,
    model,
    optimizer,
    path=TrainConfig.SAVE_PATH,
):
    # Save optimizer the state
    # https://ml-explore.github.io/mlx/build/html/python/optimizers.html
    if not os.path.exists(path):
        os.makedirs(path)

    if isinstance(optimizer, list):
        mx.eval(model.state, optimizer[0].state, optimizer[1].state)
        for iop, op in enumerate(optimizer):
            mx.save_safetensors(
                os.path.join(path, f"optimizer{iop}.safetensors"), dict(tree_flatten(op.state))
            )
    else:
        mx.eval(model.state, optimizer.state)
        mx.save_safetensors(
            os.path.join(path, "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
        )
    save_model(save_path=path, model=dequantize_model(deepcopy(model)))

    train_info = {
        "training_params": {
            **config_dict,
            # "TRAIN_DATASET_LEN": len(dataset),
        },
        "iter_step": iter_step,
        "losses": losses,
        "learning_rates": learning_rates,
        "all_rewards": all_rewards,
        "max_rewards": max_rewards,
        "std_rewards": std_rewards,
        "eval_rewards": eval_rewards
    }
    with open(os.path.join(path, "train_info.json"), "w") as f:
        json.dump(train_info, f, indent=2)
    tokenizer.save_pretrained(path)
    # print("\nModel saved", flush=True)


def interpolate_models(past_model: nn.Module, present_model: nn.Module, weight: float):
    """
    Linearly interpolate all parameters between two MLX models.
    """
    assert 0.0 <= weight <= 1.0, "weight must be in [0, 1]"
    if weight == 1:
        return present_model.parameters()
    if weight == 0:
        return past_model.parameters()

    flt_past_model = tree_flatten(past_model)
    flt_present_model = tree_flatten(present_model)
    new_weights = []

    # Iterate over all named parameters in both models
    for (psk, psw), (prk, prw) in zip(flt_past_model, flt_present_model):
        if isinstance(psw, str):
            continue
        assert psk == prk
        new_weight = psw * (1 - weight) + prw * weight
        new_weights.append((psk, new_weight))

    return tree_unflatten(new_weights)


def soft_gate(x, advantages, t_pos=1, t_neg=1.07):
    # SAPO default: t_pos=1, t_neg=1.05
    temp = mx.where(advantages > 0, t_pos, t_neg)
    return mx.sigmoid(temp * (x-1)) * (4 / temp)



def calculate_log_probs(model, io_toks, ans_toks, pad_tok_id):
    """Calculates the log probabilities of the generated answer tokens."""
    # Pass the full sequence (prompt + answer) to the model
    logits = model(io_toks)

    # Convert to log probabilities
    log_probs_full = nn.log_softmax(logits, axis=-1)

    ## Find the actual positions where answer tokens should be extracted
    # This assumes a_toks contains the actual token IDs that were generated
    batch_size, seq_len = io_toks.shape
    _, ans_len = ans_toks.shape

    # Calculate the starting position for answer tokens (assuming they're at the end)
    start_pos = seq_len - ans_len

    # Extract log probabilities for the answer portion of the sequence
    answer_log_probs = log_probs_full[:, start_pos : start_pos + ans_len, :]

    # Create indices for gathering - ensure proper shape alignment
    indices = ans_toks[:, :, None]

    # Extract log probabilities for the actual answer tokens
    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)

    # Recovery from padding
    pad_mask = mx.where(ans_toks != pad_tok_id, 1, 0)

    return selected_log_probs, pad_mask


# @partial(mx.compile, inputs=state)
def grpo_loss_fn(
    model, model_ref, io_toks, a_toks, old_logprobs, advantages, beta, pad_tok_id
):    
    """The GRPO loss function."""
    # How/why could old_logprobs matter? See: https://github.com/huggingface/trl/blob/035c3ff151b953ca72cdfe0ee966bc1469a26fde/trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py#L159
    model.train()
    # Get log probs from the trainable model (π_θ)
    logprobs, pad_mask = calculate_log_probs(model, io_toks, a_toks, pad_tok_id)
    # Get log probs from the old non-trainable model (π_θ_old)

    # PPO-clip objective
    # Ratio is converted from log values using exp(log)
    if TrainConfig.SAMPLING == 'sequence':
        # GSPO Equation: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/gspo-reinforcement-learning?q=learning+rage
        ratio = ((logprobs - old_logprobs) * pad_mask).sum(axis=-1) / pad_mask.sum(axis=-1)
        ratio = mx.exp(ratio)
        if not TrainConfig.SOFT_CLIP:
            clipped_ratio = mx.clip(ratio, 1.0 - TrainConfig.EPSILON_MIN, 1.0 + TrainConfig.EPSILON_HIGH)
            token_policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)
        else:
            token_policy_reward = soft_gate(ratio, advantages) * advantages
        # token_policy_reward.shape: (G, )
    elif TrainConfig.SAMPLING == 'token':
        # DAPO - Decoupled Clip and Dynamic sAmpling Policy Optimization: https://arxiv.org/pdf/2503.14476
        ratio = mx.exp(logprobs - old_logprobs)
        advantages = mx.expand_dims(advantages, axis=1)
        # DAPO
        if not TrainConfig.SOFT_CLIP:
            clipped_ratio = mx.clip(ratio, 1.0 - TrainConfig.EPSILON_MIN, 1.0 + TrainConfig.EPSILON_HIGH)
            token_policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages) * pad_mask
        # SAPO
        else:
            token_policy_reward = soft_gate(ratio, advantages) * advantages * pad_mask
        # token_policy_reward.shape: (G, n_tokens)
    elif TrainConfig.SAMPLING == 'custom':
        # Sequence level rewards
        token_policy_reward = advantages * ((logprobs * pad_mask).sum(axis=-1) / pad_mask.sum(axis=-1))
    else:
        NotImplementedError

    if beta > 0:
        # Get log probs from the reference model (π_ref) for KL penalty
        logprobs_ref, pad_mask = calculate_log_probs(model_ref, io_toks, a_toks, pad_tok_id)

        # KL penalty
        # Step 1: Calculate log(r) where r = π_ref / π_θ
        # log(r) = log(π_ref) - log(π_θ)
        log_ratio_for_kl = logprobs_ref - logprobs

        # Step 2: Calculate r itself by exponentiating log(r)
        # r = exp(log(r))
        ratio_for_kl = mx.exp(log_ratio_for_kl)

        # Step 3: Apply the paper's full formula: r - log(r) - 1
        kl_div = ratio_for_kl - log_ratio_for_kl - 1
        kl_div = kl_div * pad_mask
        kl_div = mx.stop_gradient(kl_div)
        
        if TrainConfig.SAMPLING == 'token':
            token_policy_reward = token_policy_reward - beta * kl_div
        else:
            kl_div = kl_div.sum(axis=-1) / pad_mask.sum(axis=-1)
            token_policy_reward = token_policy_reward - beta * kl_div

    # The objective is to maximize this, so we return the negative for minimization
    return -1 * (token_policy_reward.mean(axis=0))


# Pad sequences to the same length
def pad_sequences(sequences, pad_token_id):
    # Find the maximum length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        if isinstance(seq, list):
            seq = mx.array(seq)
        if len(seq) < max_len:
            if seq.ndim == 1:
                padding = mx.ones(shape=(max_len - len(seq)), dtype=seq.dtype) * pad_token_id
            elif seq.ndim == 2:
                r, c = seq.shape
                padding = mx.ones(shape=(max_len-r, c), dtype=seq.dtype) * pad_token_id
            padded_seq = mx.concatenate([seq, padding])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    return mx.stack(padded_sequences)



def rollout_batch(prompt, scorer, tokenizer, model, group_size):
    model.eval()
    prompt_tokens = tokenizer.encode(prompt)
    sampler = make_sampler(
        temp=TrainConfig.TEMPERATURE,
        top_p=TrainConfig.TOP_P if TrainConfig.TOP_P else 0,
        min_p=TrainConfig.MIN_P if TrainConfig.MIN_P else 0,
        top_k=TrainConfig.TOP_K if TrainConfig.TOP_K else 0
    )

    if TrainConfig.REPETITION_PENALTY and TrainConfig.REPETITION_PENALTY != 1:
        logits_processors = make_logits_processors(
            repetition_penalty=TrainConfig.REPETITION_PENALTY,
            repetition_context_size=TrainConfig.GEN_LEN
        )
    else:
        logits_processors = None
    
    rollout_tokens, rollout_rewards, rollout_a_toks, rollout_logprobs = [], [], [], []
    # Generate responses/rollouts
    for gitr in range(int(group_size * 1)):
        response_text = ""
        response_tokens = []
        response_logprob = []

        # Generate a response
        # Special Note: make sure eos_token is also generated
        for resp in stream_generate(
            model=model, 
            tokenizer=tokenizer_mlx, 
            prompt=prompt_tokens, 
            max_tokens=TrainConfig.GEN_LEN,
            sampler=sampler,
            logits_processors=logits_processors
        ):
            response_text += resp.text
            response_tokens.append(resp.token)
            response_logprob.append(resp.logprobs)
        
        # Avoiding truncated answers
        if resp.finish_reason != 'stop' or response_tokens[-1] != tokenizer.eos_token_id:
            reward = -1.0
        else:
            reward = float(scorer(response_text, False))
            # response_tokens.append(tokenizer.eos_token_id)

        # Take the logprobs only of the response tokens
        response_tokens = mx.array(response_tokens)
        response_logprob = mx.array(response_logprob)
        response_logprob = mx.take_along_axis(response_logprob, response_tokens[:, None], axis=-1).squeeze(-1)
        rollout_tokens.append(mx.concatenate([mx.array(prompt_tokens), mx.array(response_tokens)]))
        rollout_rewards.append(reward)
        rollout_a_toks.append(mx.array(response_tokens))
        rollout_logprobs.append(response_logprob)

    # Length based reward
    n_tokens = [len(toks) for toks in rollout_a_toks]
    max_tokens = max(n_tokens)
    for idx in range(len(rollout_rewards)):
        if rollout_rewards[idx] <= 0: continue
        len_norm = n_tokens[idx] / max_tokens
        rollout_rewards[idx] = 0.8 * rollout_rewards[idx] + 0.2 * len_norm

    # rollout_rewards = min_max_norm(rollout_rewards)
    if rollout_rewards:
        rrewards = mx.array(rollout_rewards)
        mean_reward = mx.mean(rrewards)
        std_reward = mx.sqrt(mx.var(rrewards))
        advantages = (rrewards - mean_reward)
        if TrainConfig.STD_NORM:
            advantages = advantages / (std_reward + 1e-12)  # Add epsilon for stability
    else:
        advantages = mx.array([])

    return rollout_tokens, rollout_a_toks, rollout_logprobs, rollout_rewards, advantages
    


def grpo_train_loop(
    model,
    tokenizer,
    optimizer,
    train_set,
    model_ref=None,
    max_iters=3000,
    group_size=8,
    batch_size=2,
    beta=0.0,
    max_ans_len=4,
    prev_iters=0,
    losses=[],
    learning_rates=[],
    all_rewards=[],
    max_rewards=[],
    std_rewards=[],
    eval_rewards=[]
):
    
    accum_grads = None
    if model_ref is not None and TrainConfig.REF_MODEL_MIXUP_ALPHA != 0:
        model_ref.update(model.parameters())
        model_ref.eval().freeze()
    
    # Create a grad function for the trainable model
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
    tot_loss = sum(losses)

    # Start training
    # if TrainConfig.TQDM:
    pbar = tqdm.tqdm(
        # range(prev_iters, max_iters, batch_size),
        total=max_iters // batch_size,
        initial=prev_iters,
        )

    # Skipped tracks when if a particular batch was skipped due to 0 std rewards
    # It avoids saving checkpoints + doing evals multiple times
    skipped = False
    while len(losses) < TrainConfig.ITERS:
        # Evaluation
        if len(losses) % TrainConfig.EVAL_STEPS == 0 and not skipped:
            eval_greedy = evaluate(model, runs=1, temp=0)
            eval_rewards.append({
                'iter': len(losses),
                'temperature': 0,
                'eval_score': sum(eval_greedy)/len(eval_greedy)
            })

            # Saving when this is the best checkpoint
            all_eval_rewards = [x['eval_score'] for x in eval_rewards]
            if max(all_eval_rewards) == all_eval_rewards[-1]:
                save_state(
                    len(losses),
                    losses,
                    learning_rates,
                    all_rewards,
                    max_rewards,
                    std_rewards,
                    eval_rewards,
                    model,
                    optimizer,
                    path=os.path.join(TrainConfig.SAVE_PATH, 'best_checkpoint'),
                )

            # Restart of not the best checkpint
            # if max([er['eval_score'] for er in eval_rewards]) != eval_rewards[-1]['eval_score']:
            #     del model
            #     mx.clear_cache()
            #     (
            #         model,
            #         optimizer,
            #         iter_step,
            #         losses,
            #         learning_rates,
            #         all_rewards,
            #         max_rewards,
            #         std_rewards,
            #         eval_rewards
            #     ) = load_state(TrainConfig.SAVE_PATH)
            #     print("Best model checkpoints restored")
            #     skipped = True
        
        # Save checkpoint
        if len(losses) % TrainConfig.SAVE_FREQ == 0 and not skipped and len(losses) > 0:
            save_state(
                len(losses),
                losses,
                learning_rates,
                all_rewards,
                max_rewards,
                std_rewards,
                eval_rewards,
                model,
                optimizer,
                path=TrainConfig.SAVE_PATH,
            )
        
        skipped = True
        # 1. Sample a batch of prompts
        # batch_indices = [bi % len(train_set) for bi in range(it, it + batch_size)]
        batch_index = random.choice(range(len(train_set)))

        # 2. Rollout: Generate G responses for each prompt using the model/old_model
        prompt = train_set[batch_index%len(train_set)]['prompt']
        scorer = train_set[batch_index]['scorer']
        rollout_tokens, rollout_a_toks, rollout_logprobs, rollout_rewards, advantages = rollout_batch(
            prompt,
            scorer, 
            tokenizer, 
            model,
            group_size
        )

        if len(rollout_rewards) < group_size // 2:
            print(f"Not enough samples to train: {[round(x, 2) for x in rollout_rewards]}. Skipping...", flush=True)
            continue
        if min(rollout_rewards) == max(rollout_rewards):
            print(f"\nNo diversity in group rewards: {[round(x, 2) for x in rollout_rewards]}. Skipping...", flush=True)
            continue

        print("\nRollouts:", [round(x, 2) for x in rollout_rewards])
        print(train_set[batch_index%len(train_set)]['prompt'])
        for re, rt in zip(rollout_rewards, rollout_a_toks):
            print(f"{re:.2f}: --> {tokenizer.decode(rt.tolist())}")
            print("---")
        print()

        rollout_tokens_padded = pad_sequences(rollout_tokens, tokenizer.pad_token_id)
        rollout_a_toks_padded = pad_sequences(rollout_a_toks, tokenizer.pad_token_id)
        rollout_logprobs = pad_sequences(rollout_logprobs, -100)

        # Optimization Step
        _loss = 0
        for _ in range(TrainConfig.NUM_MODEL_UPDATE_MU):
            loss, grads = loss_and_grad_fn(
                model=model,
                model_ref=model_ref,
                io_toks=rollout_tokens_padded,
                a_toks=rollout_a_toks_padded,
                old_logprobs=rollout_logprobs,
                advantages=advantages,
                beta=beta,
                pad_tok_id=tokenizer.pad_token_id,
            )
            if TrainConfig.GRAD_NORM is not None:
                grads, total_norm = optim.clip_grad_norm(grads, max_norm=TrainConfig.GRAD_NORM)
                del total_norm

            if TrainConfig.GRAD_ACCUM == 1:
                optimizer.update(model, grads)
                mx.eval(model, optimizer.state)
            else:
                if accum_grads is not None:
                    accum_grads = tree_map(mx.add, grads, accum_grads)
                else:
                    accum_grads = grads
                mx.eval(accum_grads)
                if len(losses) % TrainConfig.GRAD_ACCUM == 0:
                    accum_grads = tree_map(lambda g: (TrainConfig.GRAD_ACCUM / TrainConfig.BATCH_SIZE) * g, accum_grads)
                    optimizer.update(model, accum_grads)
                    mx.eval(model.parameters(), optimizer.state)
                    # print("Model weight updated")
            
            _loss += (loss.item() / TrainConfig.NUM_MODEL_UPDATE_MU)
            skipped = False

        
        # Logging
        learning_rates.append(optimizer.learning_rate.item())
        losses.append(_loss)
        tot_loss += _loss
        all_rewards.append(sum(rollout_rewards) / len(rollout_rewards))
        max_rewards.append(max(rollout_rewards))

        if TrainConfig.TQDM:
            rwds = list(map(lambda x: round(x, 2), all_rewards[-3:]))
            pbar.set_description(
                f"Loss: {tot_loss / (len(losses) + 1):.4f} | LR: {learning_rates[-1]:1.6f} | MA Score: {sum(all_rewards)/len(all_rewards):.2f} | Max: {sum(max_rewards) / len(max_rewards):.2f} | Eval: {eval_rewards[-1]['eval_score']:.2f} | Rewards: {rwds}"
            )
            pbar.update(TrainConfig.BATCH_SIZE)
        del grads, loss

        # Sync old model weights
        if TrainConfig.BETA > 0 and TrainConfig.REF_MODEL_MIXUP_ALPHA != 0:
            model_ref.update(interpolate_models(model_ref, model, TrainConfig.REF_MODEL_MIXUP_ALPHA))
            mx.eval(model_ref.state)
            model_ref.eval().freeze()
            print(f"\nIter {len(losses)+1}: Synced ref model weights.")

        if len(losses) % 5 == 0:
            prog_graph(
                len(losses),
                losses,
                learning_rates,
                all_rewards,
                eval_rewards,
                std_rewards,
                save_path=TrainConfig.SAVE_PATH,
                plot=False,
            )
            mx.clear_cache()

    # Final save of adapter weights
    save_state(
        len(losses),
        losses,
        learning_rates,
        all_rewards,
        max_rewards,
        std_rewards,
        eval_rewards,
        model,
        optimizer,
        path=os.path.join(TrainConfig.SAVE_PATH, 'final_checkpoint'),
    )

    return losses, all_rewards, max_rewards


model.unfreeze()
mx.eval(model)
print(model)

# Freeze model weights
# params = tree_flatten(model.parameters())
# freeze_keys = []
# for key, value in params:
    # print(key, value.ndim)
#     special_layers = ['embed', 'embed_tokens', 'lm_head', 'softmax', 'output', 'classifier']
#     if any(k in key for k in special_layers) or (value.ndim < 1):
#         print(key)
#         freeze_keys.append(key)
# model.freeze(recurse=True, keys=freeze_keys)

# sys.exit()

# special_layers = ['embed', 'embed_tokens', 'lm_head', 'softmax', 'output', 'classifier']
if TrainConfig.FREEZE_LAYERS:
    model.apply_to_modules(lambda k, v: v.freeze() if any(n in k for n in TrainConfig.FREEZE_LAYERS) else None)

# all_params = tree_flatten(model.parameters())
# trainable_keys = set(k for k, _ in tree_flatten(model.trainable_parameters()))

# 2. Iterate and print status
# print(f"{'Parameter Name':<40} | {'Status':<10}")
# print("-" * 55)
# for name, value in all_params:
#     status = "Trainable" if name in trainable_keys else "FROZEN"
#     print(f"{name:<40} | {status:<10}")

total_params = sum(v.size for name, v in tree_flatten(model.parameters()))
trainable_params = sum(v.size for name, v in tree_flatten(model.trainable_parameters()))

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.3f}%)")

losses, all_rewards, max_rewards = grpo_train_loop(
    model=model,
    model_ref=model_ref,
    tokenizer=tokenizer,
    optimizer=optimizer,
    train_set=train_ds,
    max_ans_len=TrainConfig.GEN_LEN,
    group_size=TrainConfig.GROUP_SIZE,
    beta=TrainConfig.BETA,
    batch_size=TrainConfig.BATCH_SIZE,
    max_iters=TrainConfig.ITERS,
    prev_iters=iter_step,
    losses=losses,
    learning_rates=learning_rates,
    all_rewards=all_rewards,
    max_rewards=max_rewards,
    std_rewards=std_rewards,
    eval_rewards=eval_rewards
)
