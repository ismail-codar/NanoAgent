# References: 
# https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb
# https://abderrahmanskiredjgithub.io/the-illustrated-grpo/The%20Illustrated%20GRPO.pdf
# https://github.com/huggingface/trl/issues/3662
# https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig
# https://towardsdatascience.com/how-to-finetune-small-language-models-to-think-with-reinforcement-learning/

import json
import os
# os.environ['HF_HUB_OFFLINE'] = '1'
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
from collections import defaultdict
import pickle
from functools import partial
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
from data.grpo.salseforce_tool import salesfores_toolcall

from data.grpo.websearch_tool import tool_calling_traces
from data.grpo.mobile_actions import mobileactions
from data.grpo.autoif import autoif_ds
from data.grpo.ifeval import ifeval_ds
from data.grpo.txt360_tool import txt360_toolcall

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


@dataclass
class TrainConfig:
    """Training configuration for GRPO."""
    # Iterations
    ITERS: int = 1_000
    GENERATE_DATA: bool = False
    BATCH_SIZE: int = 1
    GEN_LEN: int = 128 # 384
    SAVE_FREQ: int = 50
    LOAD_PREV: bool = False
    LEARNING_RATE: float = 1e-6
    WEIGHT_DECAY: float = 0.1
    EPSILON_MIN: float = 0.2      # Sequence/GSPO: 3e-4 | GRPO: 0.2 |   Note: Should not be changed
    EPSILON_HIGH: float = 0.272   # Sequence/GSPO: 4e-4 | GRPO: 0.272 | Note: Can be changed 
    GROUP_SIZE: int = 8
    WARMUP_STEPS: int = 50
    DECAY_STEPS: int = 40
    EVAL_STEPS: int = 25
    NUM_MODEL_UPDATE_MU: int = 1   # Number of backpropagations per question/rollout
    MODEL_WEIGHT_UPDATE_FREQ: int = 1 # After how many iters old_model should be synced
    GRAD_NORM: float | None = 1 #0.01
    MAX_INPUT_LEN: int = 384
    SAVE_PATH: str = "weights/NanoAgent-135M-nemotron-grpo"
    DATA_PATH: str = "data/datasets/grpo_cache.pickle"
    MODEL: str = "weights/NanoAgent-135M-nemotron-sft"
    FREEZE_LAYERS = []
    GRADIENT_CHECKPOINT_LAYERS: int | None = 6
    DYNAMIC_PADDING: bool = True
    EVAL_SAMPLES: int = 100
    TQDM: bool = True
    STD_NORM: bool = True
    TEMPERATURE: float = 0.4 # 0.8
    MIN_P: float | None = None
    TOP_K: int | None = None
    TOP_P: float = 0.9 #0.9
    REPETITION_PENALTY: float = 1.05


assert 0 < TrainConfig.GRAD_NORM or TrainConfig.GRAD_NORM is not None
assert 0 <= TrainConfig.WEIGHT_DECAY
assert 0 <= TrainConfig.MODEL_WEIGHT_UPDATE_FREQ
assert TrainConfig.GROUP_SIZE % TrainConfig.BATCH_SIZE == 0

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

if TrainConfig.MODEL_WEIGHT_UPDATE_FREQ > 1:
    model_old = load(cache_mlx_path, return_config=False)[0]
    model_old.eval().freeze()
    mx.eval(model_old.state)
else:
    model_old = None

tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-135M", add_bos=False)
assert tokenizer.pad_token_id != tokenizer.eos_token_id, "Pad token and EOS token must be different"


def linear_decay_with_warmup(
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
):
    """Learning rate scheduler with linear warmup and linear decay."""
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
    total_steps=(TrainConfig.ITERS * TrainConfig.NUM_MODEL_UPDATE_MU * TrainConfig.GROUP_SIZE) // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS * TrainConfig.NUM_MODEL_UPDATE_MU,
    decay_steps=TrainConfig.DECAY_STEPS * TrainConfig.NUM_MODEL_UPDATE_MU
)

scheduler_muon = linear_decay_with_warmup(
    base_lr=TrainConfig.LEARNING_RATE * 10 * 2, #0.001, 0.01, 0.02, 1e-3,
    total_steps=TrainConfig.ITERS // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS,
    decay_steps=TrainConfig.DECAY_STEPS
)

optimizer = optim.AdamW(
    learning_rate=scheduler, weight_decay=TrainConfig.WEIGHT_DECAY, eps=1e-12
)

# Interesting writings:
# * https://huggingface.co/blog/onekq/muon-optimizer
# * https://www.lakernewhouse.com/writing/muon-2
# * https://kellerjordan.github.io/posts/muon/
# * https://varunneal.github.io/essays/muon
# * https://github.com/KellerJordan/Muon

# optimizer = optim.MultiOptimizer(
#     [        
#         optim.Muon(
#             learning_rate=scheduler_muon, weight_decay=TrainConfig.WEIGHT_DECAY
#         ),
#         optim.AdamW(
#             learning_rate=scheduler, betas=[0.9, 0.999], weight_decay=TrainConfig.WEIGHT_DECAY, eps=1e-12
#         )
#     ],
#     # Where muon will be applied
#     [lambda name, weight: weight.ndim >= 2 and 'embed' not in name and 'norm' not in name]
# )


if TrainConfig.GENERATE_DATA:
    ds_size = 4 * TrainConfig.ITERS
    train_ds = []

    # --- IF Eval ---
    # # sz = int(ds_size * 0.4)
    # # train_ds += autoif_ds(tokenizer, TrainConfig.MAX_INPUT_LEN, n_instructions=1)
    # # train_ds = sorted(train_ds, key=lambda x: len(x['prompt']), reverse=True)#[:sz]
    # # sz = int(ds_size * 1.0)
    # train_ds += ifeval_ds(tokenizer, TrainConfig.MAX_INPUT_LEN, n_instructions=1, kshot=False)#[:sz]
    # # sz = int(ds_size * 0.4)
    # # train_ds += sorted(general_chat_ds(tokenizer, TrainConfig.MAX_INPUT_LEN), key=lambda x: len(x['prompt']), reverse=True)[:sz]
    # # random.shuffle(train_ds)
    # # train_ds = train_ds[:sz]
    # # print("IF-Eval DS size:", len(train_ds))

    
    # --- Tool Call ---
    sz = int(ds_size * 2.5)
    # train_ds += salesfores_toolcall(tokenizer, prompt_token_len=TrainConfig.MAX_INPUT_LEN, n_tool_inputs=6, dedupe_ratio=None, think=True, k_shot=False)
    train_ds += txt360_toolcall(tokenizer=tokenizer, prompt_token_len=TrainConfig.MAX_INPUT_LEN)
    # sz = int(ds_size * 0.25)
    # train_ds += tool_calling_traces(tokenizer, TrainConfig.MAX_INPUT_LEN)[:sz]
    # sz = int(ds_size * 0.05)
    # train_ds += mobileactions(tokenizer, TrainConfig.MAX_INPUT_LEN)[:sz]

    # --- Math Mix ---
    # sz = int(ds_size * 0.25)
    # train_ds += alice_in_wonderland(tokenizer=tokenizer, size=sz, think=False)
    # # sz = int(ds_size * 0.1)
    # # train_ds += syllogism(tokenizer, size=sz, think=False)
    # sz = int(ds_size * 0.3)
    # train_ds += gsm_symbolic(tokenizer, size=sz, think=False)
    # # sz = int(ds_size * 0.15)
    # # train_ds += chain_sum(tokenizer, size=sz, think=False)
    # sz = int(ds_size * 0.05) # 0.1
    # train_ds += zebra_puzzles(tokenizer, size=sz, think=False)
    # # sz = int(ds_size * 0.1)
    # # train_ds += needle_haystack(tokenizer, size=sz*3, prompt_token_len=TrainConfig.MAX_INPUT_LEN, think=False)[:sz]
    
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


def evaluate(eval_model, runs=4, temp=0):
    """Run evaluation on eval_ds and return rewards."""
    # return [0]
    if not eval_ds:
        return [0]
    # from mlx_lm.generate import generate
    rewards = []
    eval_model.eval().freeze()
    mx.eval(eval_model)

    for idx in tqdm.tqdm(range(len(eval_ds)), leave=False):
        data = eval_ds[idx]
        prompt_lead = "```json\n"
        prompt_tokens = tokenizer.encode(data['prompt'] + prompt_lead)
        scorer = data['scorer']
        for _ in range(runs):
            response = generate(
                eval_model,
                tokenizer,
                prompt_tokens,
                max_tokens=TrainConfig.GEN_LEN,
                sampler=None if temp == 0 else lambda x: mx.random.categorical(x / temp, axis=-1)
            )
            reward = scorer(prompt_lead + response, False)
            rewards.append(reward)
            
            # print(f"-- EVAL PROMPT {idx} ---")
            # print(data['prompt'])
            # print("--- EVAL GEN ---")
            # print(response)

    return rewards


def mean_map(data, win=20):
    """Compute moving average with window size win."""
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
    """Plot training progress (loss, rewards, eval scores, learning rate)."""
    plt.close()
    fig = plt.figure(figsize=(18, 12), dpi=600)
    gs = GridSpec(3, 1, height_ratios=[4, 6, 2], hspace=0.2)
    # fig.suptitle(f"GRPO Iter: {iter}", fontsize=13)

    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]

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
    # std_rewards = np.asarray(std_rewards)
    std_rewards = np.std(all_rewards)
    axes[1].plot(np.cumsum(all_rewards) / (np.arange(len(all_rewards)) + 1), color="tab:blue", alpha=0.8, linestyle=':', label='Cumulative Sum')
    axes[1].plot(mean_map(all_rewards), color="tab:blue", alpha=0.8, linestyle='--', label='Mean Win. 20')
    # axes[1].plot(gaussian_filter1d(all_rewards, sigma=2.5), linewidth=2, color="tab:blue", label='Smoothen')
    axes[1].fill_between(
        np.arange(len(all_rewards)),
        all_rewards - std_rewards,
        all_rewards + std_rewards,
        color="tab:blue",
        alpha=0.15,
        label="±1 Std"
    )
    axes[1].plot(all_rewards, alpha=1, color="tab:blue", label='Reward (batch-mean)')

    itrs = [x['iter'] for x in eval_rewards]
    eval_greedy = [e['eval_score'] for e in eval_rewards]

    axes[1].scatter(itrs, eval_greedy, color="tab:red", linewidth=3, marker="*", label="Eval Score")
    axes[1].plot(itrs, eval_greedy, color="tab:red", linewidth=1, linestyle='--')

    axes[1].set_title("Rewards")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(top=1.0)
    axes[1].set_ylim(bottom=0)

    # axes[2].scatter(itrs, eval_sampling, color="tab:green", linewidth=2, marker="x")
    # axes[2].plot(itrs, eval_sampling, color="tab:green", alpha=0.6, linestyle='--', linewidth=2)

    # axes[2].set_title("Eval Rewards (Greedy)")
    # axes[2].grid(True)

    # total_prompt_tokens
    axes[2].plot(learning_rates, color="tab:orange")
    axes[2].set_title("Learning Rate")
    axes[2].grid(True)

    # Remove [0, 1] default ticks on x-axis for bottom graph
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(
            f"{save_path}/plot.jpg", dpi=400, bbox_inches="tight", transparent=True
        )
    if plot:
        plt.show()


def load_state(path=TrainConfig.SAVE_PATH):
    """Load model, optimizer, and training state from checkpoint."""
    with open(os.path.join(path, "train_info.json"), "r") as f:
        train_info = json.load(f)
    
    model_path = os.path.join(path, "weights.safetensors")
    if os.path.exists(model_path):
        loaded_params = dict(mx.load(model_path).items())
        model.update(tree_unflatten(list(loaded_params.items())))
        mx.eval(model.state)
    else:
        model = load(Path(path), model_config=model_config)[0]
        mx.eval(model.state)
    
    optimizer_path = os.path.join(path, "optimizer.safetensors")
    if os.path.exists(optimizer_path):
        optimizer.state = tree_unflatten(
            list(mx.load(optimizer_path).items())
        )
        mx.eval(optimizer.state)

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
    """Save model, optimizer, and training state to checkpoint."""
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
    """Linearly interpolate all parameters between two MLX models."""
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


# @partial(mx.compile, inputs=model.state)
def grpo_loss_fn(
    model_train, rollout_tokens, rollout_mask, rollout_logprobs, advantages
):    
    """
    GRPO loss function.
    
    Computes policy gradient loss with optional KL penalty.
    - rollout_mask: 2=prompt (ignore), 1=response (train), 0=padding
    - advantages: normalized reward within each group
    """
    
    # Ensure arrays are MLX arrays
    rollout_tokens = mx.array(rollout_tokens) if not isinstance(rollout_tokens, mx.array) else rollout_tokens
    rollout_mask = mx.array(rollout_mask) if not isinstance(rollout_mask, mx.array) else rollout_mask
    rollout_logprobs = mx.array(rollout_logprobs) if not isinstance(rollout_logprobs, mx.array) else rollout_logprobs
    
    # Forward pass: get logits for all tokens
    logits = model_train(rollout_tokens)
    
    # Shift logits by 1 for next-token prediction (token at position t predicts token at t+1)
    log_probs = nn.log_softmax(logits, axis=-1)
    # tokens = mx.argmax(log_probs, axis=-1)
    # log_probs[t] contains probability of t+1'th token:          log_probs[t-1]:(t)   |          log_probs[t]:(t+1) | ...
    # rollout_logprobs[t] contain probability of t'th token: rollout_logprobs[t]:(t)   | rollout_logprobs[t+1]:(t+1) | ...
    log_probs = mx.roll(log_probs, shift=1, axis=-1)
    # tokens = mx.roll(tokens, shift=1, axis=-1)
    
    # Gather logprobs for the actual tokens (select prob of token_id from vocabulary distribution)
    indices = rollout_tokens[:, :, None]
    selected_log_probs = mx.take_along_axis(log_probs, indices, axis=-1).squeeze(-1)
    
    # Create binary mask: only response tokens (mask=1) contribute to loss
    response_mask = mx.where(rollout_mask == 1, 1.0, 0.0)
    log_probs_for_loss = selected_log_probs * response_mask

    # print(response_mask.shape, response_mask.shape, tokens.shape)
    # for i in range(response_mask.shape[0]):
    #     for j in range(response_mask.shape[1]):
    #         if response_mask[i][j] == 1:
    #             print(i, ':', tokenizer.decode(rollout_tokens[i, j].item()), '|', tokenizer.decode(tokens[i, j].item()))
    # input()
    
    # Broadcast advantages to per-token: same advantage for all tokens in a sequence
    advantages = mx.expand_dims(advantages, axis=1) * response_mask
    
    # === Policy gradient loss ===
    if TrainConfig.MODEL_WEIGHT_UPDATE_FREQ > 1:
        old_logprobs = mx.stop_gradient(log_probs_for_loss)
        # DAPO - Decoupled Clip and Dynamic sAmpling Policy Optimization: https://arxiv.org/pdf/2503.14476
        # Token-level GRPO: ratio = exp(new_logprob - old_logprob)
        # How/why could old_logprobs matter? See: https://github.com/huggingface/trl/blob/035c3ff151b953ca72cdfe0ee966bc1469a26fde/trl/experimental/grpo_with_replay_buffer/grpo_with_replay_buffer_trainer.py#L159
        ratio = mx.exp(log_probs_for_loss - old_logprobs)

        # Clip ratio to stabilize training (standard GRPO)
        clipped_ratio = mx.clip(ratio, 1.0 - TrainConfig.EPSILON_MIN, 1.0 + TrainConfig.EPSILON_HIGH)
        token_policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages) * response_mask

    else:
        # https://rlhfbook.com/c/06-policy-gradients
        token_policy_reward = advantages * log_probs_for_loss * response_mask

    # === Final loss normalization ===
    # Token level aggregation: verage over response tokens only
    token_policy_reward = token_policy_reward.sum() / response_mask.sum()
    # Sequence level aggregation
    # token_policy_reward = (token_policy_reward.sum(axis=-1) / response_mask.sum(axis=-1)).mean()
    
    # Negative because we maximize reward (gradient descent minimizes)
    return -1 * (token_policy_reward)


# @partial(mx.compile, inputs=model.state)
def rollout_batch(prompts, scorers, tokenizer, rollout_model, dynamic_sampling=True):
    """
    Generate rollouts for a single prompt.
    
    Two-pass dynamic padding:
    1. First loop: collect raw (unpadded) rollouts
    2. Second pass: pad all rollouts to the longest prompt+response length
    """
    rollout_model.eval().freeze()
    
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
    
    # === FIRST PASS: Collect raw rollouts without padding ===
    # raw_tokens: prompt + response (unpadded, variable length)
    # raw_masks: 2 for prompt tokens, 1 for response tokens, 0 for padding
    # raw_logprobs: 0 for prompt, actual logprobs for response tokens
    raw_tokens, raw_masks, raw_logprobs, rollout_rewards = [], [], [], []
    gen_cache = set()
    effective_group_size = TrainConfig.GROUP_SIZE // TrainConfig.BATCH_SIZE
    
    # print("--- PROMPT ---")
    # print(prompt)

    for prompt, scorer in zip(prompts, scorers):
        prompt_tokens = tokenizer.encode(prompt)
        prompt_len = len(prompt_tokens)
        for gitr in range(effective_group_size):
            response_text = ""
            response_tokens = []
            response_logprob = []

            for resp in stream_generate(
                model=rollout_model, 
                tokenizer=tokenizer_mlx, 
                prompt=prompt_tokens, 
                max_tokens=TrainConfig.GEN_LEN,
                sampler=sampler,
                logits_processors=logits_processors
            ):
                response_text += resp.text
                response_tokens.append(resp.token)
                response_logprob.append(resp.logprobs)
            
            if response_text in gen_cache:
                continue
            gen_cache.add(response_text)

            # print(f"--- RESPONSE ({len(gen_cache)}) ---")
            # print(response_text)
            
            # Reward: -1 if didn't stop properly, otherwise use scorer
            if resp.finish_reason != 'stop' or response_tokens[-1] != tokenizer.eos_token_id:
                reward = -1.0
            else:
                reward = float(scorer(response_text, False))
            # print("Reward:", reward)
            
            if dynamic_sampling and not (0 < reward < 1):
                continue

            response_tokens = mx.array(response_tokens)
            response_logprob = mx.array(response_logprob)
            response_len = len(response_tokens)
            
            # Concatenate prompt + response (no padding yet)
            full_tokens = mx.concatenate([mx.array(prompt_tokens), response_tokens])
            
            # Mask: 2 = prompt tokens (ignored in loss), 1 = response tokens (train), 0 = padding (ignore in loss)
            mask = mx.concatenate([mx.full(prompt_len, 2), mx.full(response_len, 1)])
            
            # Extract logprobs for the actual generated tokens (prompt logprobs = 0)
            logprobs_slice = mx.take_along_axis(response_logprob, response_tokens[:, None], axis=-1).squeeze(-1)
            logprobs = mx.concatenate([mx.zeros(prompt_len), logprobs_slice])
            
            raw_tokens.append(full_tokens)
            raw_masks.append(mask)
            raw_logprobs.append(logprobs)
            rollout_rewards.append(reward)
            
            if len(rollout_rewards) == effective_group_size:
                break
        if len(rollout_rewards) == effective_group_size:
            break
    
    # Handle edge case: no valid rollouts
    if not raw_tokens:
        if rollout_rewards and max(rollout_rewards) > 0:
            rrewards = mx.array(rollout_rewards)
            mean_reward = mx.mean(rrewards)
            std_reward = mx.sqrt(mx.var(rrewards))
            advantages = (rrewards - mean_reward)
            if TrainConfig.STD_NORM:
                advantages = advantages / (std_reward + 1e-12)
        else:
            advantages = mx.array([])
            rollout_rewards = []
        return [], [], [], rollout_rewards, advantages
    
    # === Find max length for dynamic padding ===
    if TrainConfig.DYNAMIC_PADDING:
        max_prompt_answer_len = max(len(t) for t in raw_tokens)
        # rem = max_prompt_answer_len % 8
        # max_prompt_answer_len += (8 - rem)
    else:
        max_prompt_answer_len = TrainConfig.MAX_INPUT_LEN + TrainConfig.GEN_LEN
        
    
    # === SECOND PASS: Pad all rollouts to max length ===
    rollout_tokens, rollout_mask, rollout_logprobs = [], [], []
    for full_tokens, mask, logprobs in zip(raw_tokens, raw_masks, raw_logprobs):
        padding_needed = max_prompt_answer_len - len(full_tokens)
        if padding_needed > 0:
            # Pad tokens with pad_token_id, mask/logprobs with zeros
            full_tokens = mx.concatenate([full_tokens, mx.full(padding_needed, tokenizer.pad_token_id)])
            mask = mx.concatenate([mask, mx.zeros(padding_needed)])
            logprobs = mx.concatenate([logprobs, mx.zeros(padding_needed)])
        
        rollout_tokens.append(full_tokens)
        rollout_mask.append(mask)
        rollout_logprobs.append(logprobs)

    if rollout_rewards and max(rollout_rewards) > 0:
        rrewards = mx.array(rollout_rewards)
        mean_reward = mx.mean(rrewards)
        std_reward = mx.sqrt(mx.var(rrewards))
        advantages = (rrewards - mean_reward)
        if TrainConfig.STD_NORM:
            advantages = advantages / (std_reward + 1e-12)
    else:
        advantages = mx.array([])
        rollout_rewards = []

    return rollout_tokens, rollout_mask, rollout_logprobs, rollout_rewards, advantages
    


def grpo_train_loop(
    model,
    model_old,
    tokenizer,
    optimizer,
    train_set,
    max_iters=3000,
    prev_iters=0,
    losses=[],
    learning_rates=[],
    all_rewards=[],
    max_rewards=[],
    std_rewards=[],
    eval_rewards=[]
):
    """Main GRPO training loop."""
    
    # Create a grad function for the trainable model
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
    tot_loss = sum(losses)

    # Start training
    # if TrainConfig.TQDM:
    pbar = tqdm.tqdm(
        # range(prev_iters, max_iters, batch_size),
        total=max_iters // TrainConfig.BATCH_SIZE,
        initial=prev_iters,
        )

    # Skipped tracks when if a particular batch was skipped due to 0 std rewards
    # It avoids saving checkpoints + doing evals multiple times
    skipped = False
    while len(losses) < TrainConfig.ITERS:
        # Evaluation
        if len(losses) % TrainConfig.EVAL_STEPS == 0 and not skipped:
            eval_temp = 0
            eval_scores = evaluate(model, runs=1, temp=eval_temp)
            eval_rewards.append({
                'iter': len(losses),
                'temperature': eval_temp,
                'eval_score': sum(eval_scores)/len(eval_scores)
            })
            print(f"\nEval Score: {sum(eval_scores)/len(eval_scores):.6f}")

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
        batch_indices = random.choices(range(len(train_set)), k=TrainConfig.GROUP_SIZE*2)

        # 2. Rollout: Generate G responses for each prompt using the model/old_model
        prompts = [train_set[bidx%len(train_set)]['prompt'] for bidx in batch_indices]
        scorers = [train_set[bidx%len(train_set)]['scorer'] for bidx in batch_indices]
        rollout_tokens, rollout_mask, rollout_logprobs, rollout_rewards, advantages = rollout_batch(
            prompts=prompts,
            scorers=scorers, 
            tokenizer=tokenizer, 
            rollout_model=model_old if TrainConfig.MODEL_WEIGHT_UPDATE_FREQ > 1 else model
        )

        if len(rollout_rewards) < TrainConfig.GROUP_SIZE:
            print(f"Not enough samples to train: {[round(x, 2) for x in rollout_rewards]}. Skipping...", flush=True)
            continue
        if min(rollout_rewards) == max(rollout_rewards):
            print(f"\nNo diversity in group rewards: {[round(x, 2) for x in rollout_rewards]}. Skipping...", flush=True)
            continue

        # print("\nRollouts:", [round(x, 2) for x in rollout_rewards])
        # # print(train_set[batch_index%len(train_set)]['prompt'])
        # # prompt_len = len(tokenizer.encode(train_set[batch_index%len(train_set)]['prompt']))
        # for re, rt in zip(rollout_rewards, rollout_tokens):
        #     response_decoded = tokenizer.decode(rt.tolist())
        #     while response_decoded.endswith('<|endoftext|>'):
        #         response_decoded = response_decoded.removesuffix('<|endoftext|>')
        #     print(f"{re:.2f}: --> {response_decoded}")
        #     print("---")
        # print()

        # Optimization Step
        _loss = 0
        for _ in range(TrainConfig.NUM_MODEL_UPDATE_MU):
            model.train().unfreeze()
            mx.eval(model)
            loss, grads = loss_and_grad_fn(
                model_train=model,
                rollout_tokens=rollout_tokens,
                rollout_mask=rollout_mask,
                rollout_logprobs=rollout_logprobs,
                advantages=advantages,
            )
            if TrainConfig.GRAD_NORM is not None:
                grads, total_norm = optim.clip_grad_norm(grads, max_norm=TrainConfig.GRAD_NORM)
                del total_norm

            optimizer.update(model, grads)
            mx.eval(model, optimizer)

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
        
        if TrainConfig.MODEL_WEIGHT_UPDATE_FREQ > 1 and len(losses) % TrainConfig.MODEL_WEIGHT_UPDATE_FREQ == 0:
            model_old.update(model.parameters())
            mx.eval(model_old.state)
            model_old.eval().freeze()
            print(f"\nIter {len(losses)+1}: Synced old model weights.")

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


model.train().unfreeze()
mx.eval(model)
# print(model)

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
    model_old=model_old,
    tokenizer=tokenizer,
    optimizer=optimizer,
    train_set=train_ds,
    max_iters=TrainConfig.ITERS,
    prev_iters=iter_step,
    losses=losses,
    learning_rates=learning_rates,
    all_rewards=all_rewards,
    max_rewards=max_rewards,
    std_rewards=std_rewards,
    eval_rewards=eval_rewards
)