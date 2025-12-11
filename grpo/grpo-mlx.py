# References: 
# https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb
# https://abderrahmanskiredjgithub.io/the-illustrated-grpo/The%20Illustrated%20GRPO.pdf
# https://github.com/huggingface/trl/issues/3662
# https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig

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

import matplotlib
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import tqdm
import random
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import generate, load
from mlx_lm.utils import load_model, save_model
from data.grpo.salseforce_tool import salesfores_toolcall
from data.grpo.websearch_tool import tool_calling_traces
from data.grpo.calculate import calculate_math
from data.grpo.gorilla_tool import gorilla_openfun
from scipy.ndimage import gaussian_filter1d

os.environ["TOKENIZERS_PARALLELISM"] = "true"
plt.ioff()
matplotlib.use("Agg")

from dataclasses import dataclass
from utils.tokenizer import get_tokenizer
from utils.webtool import tool_call_extract


@dataclass
class TrainConfig:
    # Iterations
    ITERS = 5_000
    GENERATE_DATA = False
    BATCH_SIZE = 1
    GEN_LEN = 128
    SAVE_FREQ = 50
    LOAD_PREV = False
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    EPSILON_MIN = 0.4
    EPSILON_HIGH = 0.4
    GROUP_SIZE = 4
    WARMUP_STEPS = 100
    DECAY_STEPS = 100
    BETA = 0
    UPDATE_WEIGHT = 8
    EVAL_STEPS = 50
    NUM_ITER = 1 # 1
    GRAD_NORM = 0.1
    REF_MODEL_MIXUP_ALPHA = 0.6
    MAX_INPUT_LEN = 1024 + 784
    SAVE_PATH = "weights/NanoAgent-135M-grpo-web"
    DATA_PATH = "data/datasets/grpo_unordered_cache.pickle"
    TQDM = True
    SAMPLING = "sequence"
    SOFT_CLIP = True # Soft clipping proposed in SAPO paper - https://arxiv.org/pdf/2511.20347
    TEMPERATURE = 0.9


# Token Sampling: DAPO - https://arxiv.org/pdf/2503.14476
# Sequence Sampling: GSPO - https://arxiv.org/pdf/2507.18071

assert TrainConfig.SAMPLING in ['token', 'sequence']
assert 0 <= TrainConfig.UPDATE_WEIGHT


config_dict = {
    k: v
    for k, v in TrainConfig.__dict__.items()
    if not k.startswith("__") and not callable(v)
}
print(json.dumps(config_dict, indent=2))

# The model that will be trained
MODEL_PATH = "weights/NanoAgent-135M"

model, _, model_config = load(MODEL_PATH, return_config=True)
model.train()

model_old = load(MODEL_PATH)[0]
# nn.quantize(model_old)
model_old.eval().freeze()

if TrainConfig.BETA > 0:
    # The reference model for KL-div (freezed)
    model_ref = load(MODEL_PATH)[0]
    # nn.quantize(model_ref)
    model_ref.eval().freeze()
else:
    model_ref = None

tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-135M", add_bos=False)


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
            cosine_decay,
        )

    return schedule


def linear_decay_with_warmup(
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
):
    assert total_steps - warmup_steps - decay_steps > 0
    def schedule(step):
        # Linear warmup
        warmup_lr = base_lr * step / warmup_steps
        # Linear decay
        decay_lr = base_lr * (step - (total_steps - decay_steps)) / decay_steps
        return mx.where(
            step < warmup_steps,
            warmup_lr,
            mx.where(step >= (total_steps - decay_steps), decay_lr, base_lr))

    return schedule


scheduler = cosine_decay_with_warmup(
    max_lr=TrainConfig.LEARNING_RATE,
    total_steps=TrainConfig.ITERS // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS,
)

# scheduler = linear_decay_with_warmup(
#     base_lr=TrainConfig.LEARNING_RATE,
#     total_steps=TrainConfig.ITERS // TrainConfig.BATCH_SIZE,
#     warmup_steps=TrainConfig.WARMUP_STEPS,
#     decay_steps=TrainConfig.DECAY_STEPS
# )

optimizer = optim.AdamW(
    learning_rate=scheduler, betas=[0.9, 0.99], weight_decay=TrainConfig.WEIGHT_DECAY
)


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


def total_tokens(data):
    return len(
        tokenizer.encode(
            data,
        )  # add_generation_prompt=True)
    )


def tool_tokens(ground_tool_call):
    ntokens = len(tokenizer.encode(json.dumps(ground_tool_call)))
    return ntokens


if TrainConfig.GENERATE_DATA:
    ds = tool_calling_traces(tokenizer)#+ salesfores_toolcall(tokenizer=tokenizer, n_tool_calls=2, n_tool_inputs=6, dedupe_ratio=0.95, think=False) #+ gorilla_openfun(tokenizer=tokenizer)
    # train_ds = salesfores_toolcall(tokenizer=tokenizer, n_tool_calls=1, n_tool_inputs=6, dedupe_ratio=0.95, think=False)
    ds = list(filter(lambda x: total_tokens(x['prompt']) <= TrainConfig.MAX_INPUT_LEN, ds))
    random.shuffle(ds)
    # train_ds.sort(
    #     key=lambda x: (len(json.dumps(x["ground_tool_call"])), x["num_input_tools"])
    # )
    # train_ds = train_ds[:TrainConfig.ITERS]
    train_ds = []
    cnt = defaultdict(int)
    for d in ds:
        fname = d['ground_tool_call'][0]['name']
        if cnt[fname] < 1700:
            cnt[fname] += 1
            train_ds.append(d)

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

# sys.exit()

def evaluate(eval_model, runs=2):
    # return [0]
    eval_data = train_ds[-25:]
    rewards = []
    eval_model.eval()
    for idx, data in enumerate(eval_data):
        prompt_tokens = tokenizer.encode(data['prompt'])
        scorer = data['scorer']
        for _ in range(runs):
            response = generate(
                eval_model,
                tokenizer,
                prompt_tokens,
                max_tokens=TrainConfig.GEN_LEN,
                sampler=lambda x: mx.random.categorical(x / 0.1, axis=-1)
            )
            rewards.append(scorer(llm_gen=response))
    return rewards


def mean_map(data, win=50):
    def _mean(x):
        while len(x) < win: x.append(0)
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
    save_path=None,
    plot=True,
):
    plt.close()
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), dpi=600)
    fig.suptitle(f"GRPO Iter: {iter}", fontsize=13)

    # GRPO Loss
    # axes[0].plot(np.cumsum(all_losses) / (np.arange(len(all_losses)) + 1), color="tab:red", alpha=0.6, linestyle='--')
    axes[0].plot(mean_map(all_losses), color="tab:red", alpha=0.6, linestyle='--')
    axes[0].plot(gaussian_filter1d(all_losses, sigma=2), color="tab:red", linewidth=2)
    # axes[0].plot(all_losses, color="tab:red", alpha=0.2)
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    # Rewards
    # axes[1].plot(np.cumsum(all_rewards) / (np.arange(len(all_rewards)) + 1), color="tab:blue", alpha=0.6, linestyle='--')
    axes[1].plot(mean_map(all_rewards), color="tab:blue", alpha=0.6, linestyle='--')
    axes[1].plot(gaussian_filter1d(all_rewards, sigma=2.5), linewidth=2, color="tab:blue")
    axes[1].plot(all_rewards, alpha=0.2, color="tab:blue")
    axes[1].set_title("All Reward (blue)")
    axes[1].grid(True)

    itrs = [x[0] for x in eval_rewards]
    eval_scores = [x[1] for x in eval_rewards]
    axes[2].scatter(itrs, eval_scores, color="tab:green", linewidth=2, marker="*")
    axes[2].plot(itrs, eval_scores, color="tab:green", linewidth=2)
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

tot_layers = len(model.layers)
for layer in model.layers[:3]:
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
    mx.eval(model.state, optimizer.state)
    mx.save_safetensors(
        os.path.join(path, "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )

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


# @partial(mx.compile)
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
    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(
        -1
    )

    # Recovery from padding
    pad_mask = mx.where(ans_toks != pad_tok_id, 1, 0)

    return selected_log_probs, pad_mask


def min_p_sampler(logits, min_p=0.1, temperature=0.9):
    """
    Min-p sampling for MLX.
    Args:
        logits: [vocab] MLX array of logits.
        min_p (float): threshold ∈ (0, 1]. Tokens with p >= min_p * max(p) kept.
    Returns:
        int: sampled token ID
    """
    # Softmax → probabilities
    probs = mx.softmax(logits)
    # Find maximum probability
    max_p = mx.max(probs)
    # Boolean mask: keep tokens >= min_p * max_p
    mask = probs >= (min_p * max_p)
    return mx.random.categorical((logits * mask) / temperature, axis=-1)


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


def soft_gate(x, advantages, t_pos=1, t_neg=1.05):
    temp = mx.where(advantages > 0, t_pos, t_neg)
    return mx.sigmoid(temp * (x-1)) * (4 / temp)


def grpo_loss_fn(
    model, model_old, model_ref, io_toks, a_toks, advantages, beta, pad_tok_id
):
    """The GRPO loss function."""
    # Get log probs from the trainable model (π_θ)
    log_probs, pad_mask = calculate_log_probs(model, io_toks, a_toks, pad_tok_id)
    # Get log probs from the old non-trainable model (π_θ_old)
    old_log_probs, old_pad_mask = calculate_log_probs(
        model_old, io_toks, a_toks, tokenizer.pad_token_id
    )
    old_log_probs = mx.stop_gradient(old_log_probs)

    total_tokens = pad_mask.sum()
    n_groups = io_toks.shape[0]

    # PPO-clip objective
    # Ratio is converted from log values using exp(log)
    if TrainConfig.SAMPLING == 'sequence':
        # GSPO Equation: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/gspo-reinforcement-learning?q=learning+rage
        ratio = ((log_probs - old_log_probs) * pad_mask).sum(axis=-1) / pad_mask.sum(axis=-1)
        ratio = mx.exp(ratio)
        if not TrainConfig.SOFT_CLIP:
            clipped_ratio = mx.clip(ratio, 1.0 - TrainConfig.EPSILON_MIN, 1.0 + TrainConfig.EPSILON_HIGH)
            token_policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)
        else:
            token_policy_reward = soft_gate(ratio, advantages) * advantages
        # token_policy_reward.shape: (G, )
    elif TrainConfig.SAMPLING == 'token':
        # DAPO - Decoupled Clip and Dynamic sAmpling Policy Optimization: https://arxiv.org/pdf/2503.14476
        ratio = mx.exp(log_probs - old_log_probs)
        advantages = mx.expand_dims(advantages, 1)
        # DAPO
        if not TrainConfig.SOFT_CLIP:
            clipped_ratio = mx.clip(ratio, 1.0 - TrainConfig.EPSILON_MIN, 1.0 + TrainConfig.EPSILON_HIGH)
            token_policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages) * pad_mask
        # SAPO
        else:
            token_policy_reward = soft_gate(ratio, advantages) * advantages * pad_mask
        # token_policy_reward.shape: (G, n_tokens)
    else:
        NotImplementedError

    if beta > 0:
        # Get log probs from the reference model (π_ref) for KL penalty
        log_probs_ref, pad_mask = calculate_log_probs(model_ref, io_toks, a_toks, pad_tok_id)

        # KL penalty
        # Step 1: Calculate log(r) where r = π_ref / π_θ
        # log(r) = log(π_ref) - log(π_θ)
        log_ratio_for_kl = log_probs_ref - log_probs

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
            kl_div = kl_div.sum(axis=-1)
            token_policy_reward = token_policy_reward - beta * kl_div

    # The objective is to maximize this, so we return the negative for minimization
    if TrainConfig.SAMPLING == 'token':
        loss = -1 * ((token_policy_reward.sum() / total_tokens) / n_groups)
    else:
        loss = -1 * (token_policy_reward.sum() / n_groups)
    return loss


# Pad sequences to the same length
def pad_sequences(sequences, pad_token_id):
    if not sequences:
        return mx.array([])

    # Find hte maximum length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        if len(seq) < max_len:
            padding = mx.array([pad_token_id] * (max_len - len(seq)))
            padded_seq = mx.concatenate([seq, padding])

        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    return mx.stack(padded_sequences)

def grpo_train_loop(
    model,
    model_old,
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
    
    model.train()
    if model_old is not None:
        model_old.update(model.parameters())
        model_old.eval().freeze()
    if model_ref is not None:
        model_ref.update(model.parameters())
        model_ref.eval().freeze()
    
    # Create a grad function for the trainable model
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
    tot_loss = sum(losses)

    # Start training
    if TrainConfig.TQDM:
        pbar = tqdm.tqdm(
            range(prev_iters, max_iters, batch_size),
            total=max_iters // batch_size,
            initial=prev_iters,
        )
    else:
        pbar = range(prev_iters, max_iters, batch_size)

    # Epoch loop
    for it in pbar:
        # Evaluation
        if it % TrainConfig.EVAL_STEPS == 0:
            eval_scores = evaluate(model)
            eval_rewards.append([it, sum(eval_scores)/len(eval_scores)])
        model.train()

        # 1. Sample a batch of prompts
        batch_indices = [bi % len(train_set) for bi in range(it, it + batch_size)]

        # 2. Rollout: Generate G responses for each prompt using the old model
        rollout_tokens = []
        rollout_rewards = []
        rollout_a_toks = []

        # Batch loop
        for i in batch_indices:
            prompt_tokens = tokenizer.encode(train_set[i%len(train_set)]['prompt'])
            scorer = train_set[i]['scorer']
            group_rewards = []
            rollouts = []

            # Generate responses/rollouts
            for gitr in range(group_size * 2):
                # Generate a response
                response = generate(
                    model_old,
                    tokenizer,
                    prompt_tokens,
                    max_tokens=max_ans_len,
                    sampler=lambda x: mx.random.categorical(x / TrainConfig.TEMPERATURE, axis=-1),
                )

                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                # Avoiding truncated answers
                if len(response_tokens) >= max_ans_len - 1:
                    continue

                # Embedding EOS token as model.generate removes it
                response_tokens.append(tokenizer.eos_token_id)
                reward = scorer(llm_gen=response)
                
                full_sequence = mx.array(prompt_tokens + response_tokens)
                rollouts.append((reward, full_sequence, mx.array(response_tokens)))

            # Sort rewards hi to low
            rollouts = sorted(rollouts, key=lambda x: x[0], reverse=True)
            # DAPO: Pick rewards where we see a good reward distribution shift
            valid_rollout_indices = []
            unq_rewards = set()
            for ridx, (re, fs, rt) in enumerate(rollouts):
                if re not in unq_rewards:
                    valid_rollout_indices.append(ridx)
                    unq_rewards.add(re)
                if len(valid_rollout_indices) == TrainConfig.GROUP_SIZE:
                    break

            # Remove -1 rewards if possible
            if len(valid_rollout_indices) > 2 and -1 in unq_rewards:
                valid_rollout_indices.pop()

            if len(valid_rollout_indices) <= 1:
                print(f"\nNo diversity in group rewards: {[x[0] for x in rollouts]}. Skipping...")
                continue

            rollouts = [rollouts[p] for p in valid_rollout_indices]
            print("\nRollouts:", [x[0] for x in rollouts])
            # Store data for the optimization step
            group_rewards.extend([x[0] for x in rollouts])
            rollout_tokens.extend([x[1] for x in rollouts])
            rollout_a_toks.extend([x[2] for x in rollouts])
            
            all_rewards.append(np.mean(group_rewards).item())
            max_rewards.append(max(group_rewards))
            rollout_rewards.append(mx.array(group_rewards))

        if not rollout_rewards:
            # print("Empty rollout rewards. Skipping...", flush=True)
            continue

        # Compute Advantages
        advantages = []
        for rewards in rollout_rewards:
            mean_reward = mx.mean(rewards)
            std_reward = mx.sqrt(mx.var(rewards)) + 1e-8  # Add epsilon for stability
            adv = (rewards - mean_reward) / std_reward
            std_rewards.append(std_reward.item())
            advantages.append(adv)

        advantages = mx.concatenate(advantages)
        rollout_tokens_padded = pad_sequences(rollout_tokens, tokenizer.pad_token_id)
        rollout_a_toks_padded = pad_sequences(rollout_a_toks, tokenizer.pad_token_id)

        # Optimization Step
        _loss = 0
        for _ in range(TrainConfig.NUM_ITER):
            loss, grads = loss_and_grad_fn(
                model=model,
                model_old=model_old,
                model_ref=model_ref,
                io_toks=rollout_tokens_padded,
                a_toks=rollout_a_toks_padded,
                advantages=advantages,
                beta=beta,
                pad_tok_id=tokenizer.pad_token_id,
            )
            clipped_grads, total_norm = optim.clip_grad_norm(grads, max_norm=TrainConfig.GRAD_NORM)
            optimizer.update(model, clipped_grads)
            mx.eval(model.parameters(), optimizer.state)
            _loss += - (loss.item() / TrainConfig.NUM_ITER)

        losses.append(_loss)
        learning_rates.append(optimizer.learning_rate.item())
        tot_loss += _loss

        if TrainConfig.TQDM:
            rwds = list(map(lambda x: round(x, 2), all_rewards[-3:]))
            pbar.set_description(
                f"Loss: {losses[-1]:.4f} | {tot_loss / (it + 1):.4f} | LR: {optimizer.learning_rate.item():1.6f} | MA Score: {sum(all_rewards)/len(all_rewards):.2f} | Max {sum(max_rewards) / len(max_rewards):.2f} | Eval: {eval_rewards[-1][-1]:.2f} | Rewards: {rwds}"
            )
        del grads, clipped_grads, total_norm, loss

        # Sync old model weights
        if TrainConfig.UPDATE_WEIGHT >= 1 and it % TrainConfig.UPDATE_WEIGHT == 0:
            model_old.update(model.parameters())
            # nn.quantize(model_old, bits=8)
            mx.eval(model_old)
            model_old.eval().freeze()
            print(f"\nIter {it+1}: Synced old model weights.")
        elif TrainConfig.UPDATE_WEIGHT < 1:
            model_old.update(interpolate_models(model_old, model, TrainConfig.UPDATE_WEIGHT))
            mx.eval(model_old)
            model_old.eval().freeze()
        
        if TrainConfig.BETA > 0:
            # model_ref.update(model.parameters())
            # nn.quantize(model_ref, bits=8)
            model_ref.update(interpolate_models(model_ref, model, TrainConfig.REF_MODEL_MIXUP_ALPHA))
            mx.eval(model_ref)
            model_ref.eval().freeze()
            # print(f"\nIter {it+1}: Synced ref model weights.")


        if it % TrainConfig.SAVE_FREQ == 0:
            save_state(
                it,
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

        if it % 10 == 0:
            prog_graph(
                it,
                losses,
                learning_rates,
                all_rewards,
                eval_rewards,
                save_path=TrainConfig.SAVE_PATH,
                plot=False,
            )
            mx.clear_cache()

    # Final save of adapter weights
    model.save_weights("adapters.safetensors")
    print("Saved final weights to adapters/adapters.safetensors.")
    return losses, all_rewards, max_rewards


losses, all_rewards, max_rewards = grpo_train_loop(
    model=model,
    model_old=model_old,
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
