# ============================================================
# MiniLLM Version 2 Prototype
# - teacher
# - student
# - mixed sampling
# - single-step reverse-KL term only
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from rouge_metric import compute_metrics
from itertools import cycle


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class Config:
    # -------------------------
    # Data
    # -------------------------
    train_file: str = "dolly-processed/full/gpt2/train.jsonl"
    valid_file: str = "dolly-processed/full/gpt2/valid.jsonl"

    # -------------------------
    # Models
    # -------------------------
    teacher_model_name: str = "MiniLLM/teacher-gpt2-1.5B"
    student_init_path: str = "outputs/gpt2_small_dolly_best_SFT_lr_5e-5"

    # -------------------------
    # Output
    # -------------------------
    output_dir: str = "outputs/minillm_phase2_gpt2_V2"
    best_model_dir: str = "outputs/minillm_phase2_gpt2_best_V2"

    # -------------------------
    # Training
    # -------------------------
    seed: int = 42
    lr: float = 5e-6
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_steps: int = 25
    weight_decay: float = 0.01

    # -------------------------
    # Prompt / rollout
    # -------------------------
    max_prompt_length: int = 256
    max_new_tokens: int = 16
    max_total_length: int = 512
    temperature: float = 1.0
    alpha: float = 0.5

    # -------------------------
    # PPO / rollout optimization
    # -------------------------
    gamma: float = 1.0
    cliprange: float = 0.2
    cliprange_reward: Optional[float] = None
    whiten_advantages: bool = True

    single_step_reg: bool = True
    single_step_reg_coef: float = 0.05

    num_rollout_batches: int = 32  # 128 * batch_size(2) = 256 sampled sequences
    ppo_epochs: int = 4
    teacher_mixed_alpha: Optional[float] = 0.5

    # -------------------------
    # LM + KD
    # -------------------------
    kd_ratio: Optional[float] = 0.5
    lm_coef: float = 0.3

    # -------------------------
    # Logging / saving / eval
    # -------------------------
    log_every: int = 1
    eval_every: int = 5
    save_every: int = 50

    eval_max_new_tokens: int = 100
    num_eval_samples: Optional[int] = 200  # use None for full validation set


CFG = Config()


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# ============================================================
# 4. Dataset loading
# ============================================================

def load_dolly_dataset(train_file: str, valid_file: str):
    data_files = {
        "train": train_file,
        "validation": valid_file,
    }
    ds = load_dataset("json", data_files=data_files)
    print(ds)
    return ds



# ============================================================
# 1. Build prompt text
# ============================================================

def build_prompt(example):
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_text = str(example.get("output", "")).strip()

    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"

    return {
        "prompt_text": prompt,
        "reference_text": output_text,
    }


# ============================================================
# 2. Preprocess dataset
# ============================================================

def preprocess_dataset(ds):
    ds = ds.map(build_prompt)

    ds = ds.remove_columns(
        [c for c in ds["train"].column_names if c not in ["prompt_text", "reference_text"]]
    )

    print("\nSample prompt:")
    print(ds["train"][0]["prompt_text"])
    print("\nSample reference:")
    print(ds["train"][0]["reference_text"])

    return ds


# ============================================================
# 3. Tokenizer
# ============================================================

def load_tokenizer(model_name_or_path: str):
    print(f"\nLoading tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


# ============================================================
# 4. PPo collate from text dataset
# ============================================================

def prompt_response_collate_fn(batch, tokenizer, max_prompt_length, max_length):
    prompts = [x["prompt_text"] for x in batch]
    references = [x["reference_text"] for x in batch]

    bs = len(batch)
    pad_id = tokenizer.eos_token_id

    # -----------------------------
    # A. prompt-only batch (like model_batch in reviewed code)
    # -----------------------------
    prompt_enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )

    model_batch = {
        "input_ids": prompt_enc["input_ids"],           # left padded
        "attention_mask": prompt_enc["attention_mask"], # left padded mask
    }

    # -----------------------------
    # B. full prompt+response batch (like no_model_batch in reviewed code)
    # -----------------------------
    no_model_batch = {
        "full_ids": torch.ones(bs, max_length, dtype=torch.long) * pad_id,
        "full_attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
        "full_label_ids": torch.ones(bs, max_length, dtype=torch.long) * -100,
    }

    for i, (prompt_text, response_text) in enumerate(zip(prompts, references)):
        # tokenized prompt only
        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )

        # tokenized response only
        response_ids = tokenizer.encode(
            response_text,
            add_special_tokens=False,
        )

        # full sequence = prompt + response
        full_ids = prompt_ids + response_ids

        # keep within max_length
        full_ids = full_ids[:max_length]

        # actual usable sequence for autoregressive training:
        # input side uses full_ids[:-1], label side predicts full_ids[1:]
        input_len = len(full_ids)

        if input_len <= 1:
            continue

        # fill full_ids[:-1]
        no_model_batch["full_ids"][i, :input_len - 1] = torch.tensor(full_ids[:-1], dtype=torch.long)
        no_model_batch["full_attention_mask"][i, :input_len - 1] = 1

        # label only on response region
        # response starts after prompt_ids, but because labels are shifted by 1,
        # the first response token is predicted at position len(prompt_ids)-1
        response_start = max(len(prompt_ids) - 1, 0)

        # labels correspond to full_ids[1:]
        label_ids = full_ids[1:]
        no_model_batch["full_label_ids"][i, :len(label_ids)] = -100
        no_model_batch["full_label_ids"][i, response_start:len(label_ids)] = torch.tensor(
            label_ids[response_start:], dtype=torch.long
        )

    return model_batch, no_model_batch


# ============================================================
# 5. Dataloader builder
# ============================================================

def build_prompt_dataloader(
    dataset_split,
    tokenizer,
    batch_size,
    max_prompt_length,
    max_length,
    shuffle=False,
):
    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: prompt_response_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
        ),
    )
# ============================================================
# 4. LM collate function
# ============================================================

def lm_collate_fn(batch, tokenizer, max_length, model_type="gpt2"):
    prompts = [x["prompt_text"] for x in batch]
    references = [x["reference_text"] for x in batch]

    bs = len(batch)
    pad_id = tokenizer.eos_token_id

    model_batch = {
        "input_ids": torch.ones(bs, max_length, dtype=torch.long) * pad_id,
        "attention_mask": torch.zeros(bs, max_length, dtype=torch.long),
    }

    if model_type in ["gpt2"]:
        model_batch["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)

    no_model_batch = {
        "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
        "loss_mask": torch.zeros(bs, max_length, dtype=torch.float32),
    }

    for i, (prompt_text, response_text) in enumerate(zip(prompts, references)):
        # tokenize prompt only
        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
        )

        # tokenize response only
        response_ids = tokenizer.encode(
            response_text,
            add_special_tokens=False,
        )

        # full sequence = prompt + response
        full_ids = (prompt_ids + response_ids)[:max_length]
        input_len = len(full_ids)

        if input_len <= 1:
            continue

        # model input is shifted right
        model_batch["input_ids"][i, :input_len - 1] = torch.tensor(full_ids[:-1], dtype=torch.long)
        model_batch["attention_mask"][i, :input_len - 1] = 1

        if model_type in ["gpt2"]:
            model_batch["position_ids"][i, :input_len - 1] = torch.arange(0, input_len - 1, dtype=torch.long)

        # labels are next-token targets
        labels = full_ids[1:]
        no_model_batch["label"][i, :input_len - 1] = torch.tensor(labels, dtype=torch.long)
        no_model_batch["loss_mask"][i, :input_len - 1] = 1.0

        # mask out prompt region from LM loss
        # first response token is predicted at position len(prompt_ids)-1
        source_len = min(len(prompt_ids), max_length)
        prompt_mask_end = max(source_len - 1, 0)

        if prompt_mask_end > 0:
            no_model_batch["label"][i, :prompt_mask_end] = -100
            no_model_batch["loss_mask"][i, :prompt_mask_end] = 0.0

    return model_batch, no_model_batch


# ============================================================
# 5. LM dataloader builder
# ============================================================

def build_lm_dataloader(
    dataset_split,
    tokenizer,
    batch_size,
    max_length,
    model_type="gpt2",
    shuffle=False,
):
    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: lm_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            max_length=max_length,
            model_type=model_type,
        ),
    )
#============================================================
#  PPO Model
#===========================================================

class PPOModel(nn.Module):
    def __init__(self, model_path: str, device: torch.device):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.base_model.to(device)
        self.base_model.eval()  # no dropout for RL-style updates

    def forward(self, **x):
        return self.base_model(**x)

    def generate(self, **x):
        return self.base_model.generate(**x)

# ===========================================================
# reward computation
# ===========================================================

class Reward:
    def __init__(self, cfg, tokenizer, model):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def get_input_batch(
        self,
        input_ids: torch.Tensor,
        gen_ids: torch.Tensor,
        output_pos: bool = True,
    ):
        full_ids = torch.cat([input_ids, gen_ids], dim=-1)
        attention_mask = (full_ids != self.pad_token_id).long()

        model_inputs = {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }

        if output_pos:
            position_ids = torch.cumsum(attention_mask, dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            model_inputs["position_ids"] = position_ids

        return model_inputs

    @torch.no_grad()
    def reward_fn(
        self,
        input_ids: torch.Tensor,
        gen_ids: torch.Tensor,
        inf_mask: Optional[torch.Tensor] = None,
        output_pos: bool = True,
    ):
        self.model.eval()

        model_inputs = self.get_input_batch(
            input_ids=input_ids,
            gen_ids=gen_ids,
            output_pos=output_pos,
        )

        outputs = self.model(**model_inputs)
        logits = outputs.logits  # [B, L, V]

        # center logits, same idea as reviewed code
        logits = logits - torch.mean(logits, dim=-1, keepdim=True)

        mask = model_inputs["attention_mask"]
        logits = logits * mask.unsqueeze(-1)

        # keep positions aligned with generated tokens
        logits = logits[:, input_ids.size(-1) - 1 :, :]
        mask = mask[:, input_ids.size(-1) - 1 :]

        # score the actually generated token at each step
        selected_logits = torch.gather(
            logits[:, :-1, :],
            dim=-1,
            index=model_inputs["input_ids"][:, input_ids.size(-1) :].unsqueeze(-1),
        ).squeeze(-1)

        next_state_value = torch.logsumexp(logits[:, :-1, :], dim=-1)
        next_state_value = next_state_value * mask[:, :-1]

        scores = selected_logits - next_state_value

        assert not torch.isnan(scores).any(), "NaN in reward scores"
        assert not torch.isinf(scores).any(), "Inf in reward scores"
        assert scores.shape == gen_ids.shape, f"Expected {gen_ids.shape}, got {scores.shape}"

        return {
            "rewards": scores,
            "inf_mask": inf_mask,
        }


# ============================================================
# Rollout container
# ============================================================

@dataclass
class RolloutBatch:
    query_ids: torch.Tensor              # [B, prompt_len]
    response_ids: torch.Tensor           # [B, gen_len]
    full_ids: torch.Tensor               # [B, prompt_len + gen_len]
    full_attention_mask: torch.Tensor    # [B, prompt_len + gen_len]
    generated_token_mask: torch.Tensor   # [B, gen_len]
    teacher_rewards: torch.Tensor        # [B, gen_len]
    student_logprobs: torch.Tensor 
    w: torch.Tensor      # [B, gen_len]


# ============================================================
# Helper: build position_ids for GPT-style models
# ============================================================

def build_position_ids(attention_mask: torch.Tensor):
    position_ids = torch.cumsum(attention_mask, dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


# ============================================================
# Helper: compute student logprobs on generated tokens
# ============================================================

@torch.no_grad()
def compute_student_logprobs(student, full_ids, full_attention_mask, response_ids, temperature=1.0):
    model_inputs = {
        "input_ids": full_ids,
        "attention_mask": full_attention_mask,
        "position_ids": build_position_ids(full_attention_mask),
        "use_cache": False,
    }

    outputs = student(**model_inputs)
    logits = outputs.logits / temperature  # [B, L, V]

    prompt_width = full_ids.size(1) - response_ids.size(1)

    # We need the logits that predict each generated token
    # token 0 of response is predicted from position prompt_width-1
    gen_logits = logits[:, prompt_width - 1 : prompt_width - 1 + response_ids.size(1), :]  # [B, gen_len, V]

    log_probs = F.log_softmax(gen_logits, dim=-1)
    student_logprobs = torch.gather(
        log_probs,
        dim=-1,
        index=response_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, gen_len]

    return student_logprobs


# ============================================================
# Helper: build generated-token mask
# ============================================================

def build_generated_token_mask(response_ids, pad_token_id):
    # valid generated tokens are all non-pad tokens
    return (response_ids != pad_token_id).float()

# ============================================================
# Helper: whiten
# ============================================================

def whiten(xs: torch.Tensor, shift_mean: bool = True, distributed: bool = False) -> torch.Tensor:
    """Whitens values"""
    var, mean = torch.var_mean(xs)
    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


# ============================================================
# Helper:
# ============================================================

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_teacher_mixed(
    student,
    teacher,
    input_ids,
    attention_mask,
    tokenizer,
    max_new_tokens,
    alpha=0.5,
    temperature=1.0,
):
    device = input_ids.device
    cur_ids = input_ids.clone()
    cur_mask = attention_mask.clone()

    sampled_tokens = []
    mixed_token_logprobs = []
    student_token_logprobs = []
    teacher_token_logprobs = []

    for _ in range(max_new_tokens):
        # build position ids
        position_ids = torch.cumsum(cur_mask, dim=-1) - 1
        position_ids.masked_fill_(cur_mask == 0, 0)

        # student forward
        s_out = student(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        s_logits = s_out.logits[:, -1, :] / temperature
        s_logprobs = F.log_softmax(s_logits, dim=-1)
        s_probs = torch.exp(s_logprobs)

        # teacher forward
        t_out = teacher(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        t_logits = t_out.logits[:, -1, :] / temperature
        t_logprobs = F.log_softmax(t_logits, dim=-1)
        t_probs = torch.exp(t_logprobs)

        # mixture distribution
        mix_probs = (1.0 - alpha) * s_probs + alpha * t_probs
        mix_probs = mix_probs / mix_probs.sum(dim=-1, keepdim=True)

        # sample next token from mixed distribution
        next_token = torch.multinomial(mix_probs, num_samples=1)  # [B, 1]

        # logprobs of sampled token under each distribution
        mix_logprob = torch.log(torch.gather(mix_probs, 1, next_token).clamp_min(1e-12)).squeeze(-1)
        s_logprob = torch.gather(s_logprobs, 1, next_token).squeeze(-1)
        t_logprob = torch.gather(t_logprobs, 1, next_token).squeeze(-1)

        sampled_tokens.append(next_token)
        mixed_token_logprobs.append(mix_logprob)
        student_token_logprobs.append(s_logprob)
        teacher_token_logprobs.append(t_logprob)

        # append token
        cur_ids = torch.cat([cur_ids, next_token], dim=-1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(cur_mask.size(0), 1, dtype=cur_mask.dtype, device=device)],
            dim=-1,
        )

        # optional EOS stop if all sequences ended
        if (next_token.squeeze(-1) == tokenizer.eos_token_id).all():
            break

    if len(sampled_tokens) == 0:
        B = input_ids.size(0)
        empty = torch.empty(B, 0, dtype=torch.long, device=device)
        emptyf = torch.empty(B, 0, dtype=torch.float32, device=device)
        return {
            "full_ids": cur_ids,
            "response_ids": empty,
            "student_logprobs": emptyf,
            "teacher_logprobs": emptyf,
            "mixed_logprobs": emptyf,
            "w": emptyf,
        }

    response_ids = torch.cat(sampled_tokens, dim=-1)
    s_lp = torch.stack(student_token_logprobs, dim=1)
    t_lp = torch.stack(teacher_token_logprobs, dim=1)
    mix_lp = torch.stack(mixed_token_logprobs, dim=1)

    # importance weight per token: student_prob / mixed_prob
    w = torch.exp(s_lp - mix_lp)

    return {
        "full_ids": cur_ids,
        "response_ids": response_ids,
        "student_logprobs": s_lp,
        "teacher_logprobs": t_lp,
        "mixed_logprobs": mix_lp,
        "w": w,
    }


# ============================================================
# Sampler for one prompt batch
# ============================================================

@torch.no_grad()
def collect_rollout_batch(student, reward, model_batch, tokenizer, cfg, device, teacher=None ):
    """
    model_batch must contain only:
        - input_ids
        - attention_mask
    """

    query_ids = model_batch["input_ids"].to(device)
    attention_mask = model_batch["attention_mask"].to(device)
    if teacher is not None and getattr(cfg, "teacher_mixed_alpha", None) is not None:
        gen = generate_teacher_mixed(
            student=student,
            teacher=teacher,
            input_ids=query_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            alpha=cfg.teacher_mixed_alpha,
            temperature=cfg.temperature,
        )
        full_ids = gen["full_ids"]
        response_ids = gen["response_ids"]
        student_logprobs = gen["student_logprobs"]
        w = gen["w"]
    else:

    # generate with student
        full_ids = student.generate(
            input_ids=query_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # build full attention mask
        full_attention_mask = (full_ids != tokenizer.pad_token_id).long()

        # split prompt / response
        response_ids = full_ids[:, query_ids.size(1):]
        # student logprobs on the generated tokens
        student_logprobs = compute_student_logprobs(
            student=student,
            full_ids=full_ids,
            full_attention_mask=full_attention_mask,
            response_ids=response_ids,
            temperature=cfg.temperature,
        )
        w = torch.ones_like(student_logprobs)

    full_attention_mask = (full_ids != tokenizer.pad_token_id).long()

    # mask for valid generated tokens
    generated_token_mask = build_generated_token_mask(
        response_ids=response_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    # teacher reward
    reward_out = reward.reward_fn(
        input_ids=query_ids,
        gen_ids=response_ids,
        inf_mask=None,
        output_pos=True,
    )
    teacher_rewards = reward_out["rewards"]

    
    # zero out padded positions to keep things clean
    teacher_rewards = teacher_rewards * generated_token_mask
    student_logprobs = student_logprobs * generated_token_mask
    w = w * generated_token_mask


    return RolloutBatch(
        query_ids=query_ids,
        response_ids=response_ids,
        full_ids=full_ids,
        full_attention_mask=full_attention_mask,
        generated_token_mask=generated_token_mask,
        teacher_rewards=teacher_rewards,
        student_logprobs=student_logprobs,
        w=w
    )


# ============================================================

# ============================================================


class MiniSampler:
    def __init__(self, student, teacher, reward, prompt_loader, tokenizer, cfg, device):
        self.student = student
        self.teacher=teacher
        self.reward = reward
        self.prompt_loader = prompt_loader
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.iterator = iter(self.prompt_loader)
        self.epochs = 0

    def sample_one(self):
        try:
            model_batch, _ = next(self.iterator)
        except StopIteration:
            self.epochs += 1
            self.iterator = iter(self.prompt_loader)
            model_batch, _ = next(self.iterator)

        rollout = collect_rollout_batch(
            student=self.student,
            teacher=self.teacher,
            reward=self.reward,
            model_batch=model_batch,
            tokenizer=self.tokenizer,
            cfg=self.cfg,
            device=self.device,
        )
        return rollout

    def sample_many(self, num_rollouts):
        rollouts = []
        while len(rollouts) < num_rollouts:
            rollouts.append(self.sample_one())
        return rollouts
    
class MiniRolloutStore:
    def __init__(self):
        self.history = []

    def clear(self):
        self.history = []

    def push(self, rollouts):
        self.history.extend(rollouts)

    def __len__(self):
        return len(self.history)
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniLLMLoss:
    def __init__(self, cfg, tokenizer, teacher):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.pad_token_id = tokenizer.pad_token_id
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    # ------------------------------------------------------------
    # Helpers from reviewed logic
    # ------------------------------------------------------------
    def _build_position_ids(self, attention_mask: torch.Tensor):
        position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    def _compute_response_logits(self, model, full_ids, full_attention_mask, response_ids):
        model_inputs = {
            "input_ids": full_ids,
            "attention_mask": full_attention_mask,
            "position_ids": self._build_position_ids(full_attention_mask),
            "use_cache": False,
            "return_dict": True,
        }

        outputs = model(**model_inputs)
        logits = outputs.logits / self.cfg.temperature  # [B, L, V]

        prompt_width = full_ids.size(1) - response_ids.size(1)
        response_logits = logits[:, prompt_width - 1 : prompt_width - 1 + response_ids.size(1), :]
        return response_logits

    def _compute_token_logprobs(self, logits, target_ids):
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        return token_logprobs

    
    def _whiten_masked(self, x, mask):
        whitened = x.clone()
        valid = mask.bool()
        whitened_valid = whiten(x[valid], shift_mean=True, distributed=False)
        whitened[valid] = whitened_valid
        whitened[~valid] = 0.0
        return whitened

    # ------------------------------------------------------------
    # Reviewed logic: cumulative rewards
    # ------------------------------------------------------------
    def _get_cumsum_rewards(self, rewards):
        full_rewards = torch.zeros_like(rewards[:, 0])
        for t in reversed(range(rewards.size(1))):
            full_rewards = self.cfg.gamma * full_rewards + rewards[:, t]
        return full_rewards

    # ------------------------------------------------------------
    # Reviewed logic: advantages
    # ------------------------------------------------------------
    def _get_advantages_and_returns(
        self,
        rewards: torch.Tensor,          # [B, T]
        response_length: int,
        mask: torch.Tensor,             # [B, T]
        use_whitening: bool = True,
    ):
        last_rw = 0
        rw_reversed = []

        rewards = rewards.float()
        mask = mask.float()

        lens = torch.cumsum(mask, dim=-1)
        lens = mask - lens + lens[:, -1:None]
        lens = torch.masked_fill(lens, lens == 0, 1)

        for t in reversed(range(response_length)):
            rw_delta = rewards[:, t]
            last_rw = rw_delta + self.cfg.gamma * last_rw
            rw_reversed.append(last_rw)

        rw = torch.stack(rw_reversed[::-1], dim=1)
        rw = rw / lens

        advantages = rw

        if use_whitening:
            advantages = self._whiten_masked(advantages, mask)

        return advantages.detach()

    # ------------------------------------------------------------
    # Reviewed logic: PPO clipped policy loss
    # ------------------------------------------------------------
    def _pg_loss(
        self,
        logprobs: torch.Tensor,         # current logprobs [B, T]
        old_logprobs: torch.Tensor,     # rollout-time logprobs [B, T]
        advantages: torch.Tensor,       # [B, T]
        mask: torch.Tensor,             # [B, T]
        w: torch.Tensor,                # [B, T]
    ):
        n = mask.sum().clamp_min(1.0)

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio.float())
        ratio = ratio * w

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cfg.cliprange,
            1.0 + self.cfg.cliprange,
        )

        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2).float() * mask) / n
        return pg_loss

    # ------------------------------------------------------------
    # Reviewed logic: regularization loss
    # ------------------------------------------------------------
    def _reg_loss(self, rollout, student_logits, mask):
        with torch.no_grad():
            teacher_logits = self._compute_response_logits(
                model=self.teacher,
                full_ids=rollout.full_ids,
                full_attention_mask=rollout.full_attention_mask,
                response_ids=rollout.response_ids,
            )

        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)

        kd_per_token = -(teacher_probs * student_log_probs).sum(dim=-1)
        reg_loss = (kd_per_token * mask).sum() / mask.sum().clamp_min(1.0)
        return reg_loss

    # ------------------------------------------------------------
    # PPO / RL loss for your RolloutBatch
    # ------------------------------------------------------------
    def rollout_loss(self, student, rollout):
        mask = rollout.generated_token_mask.float()               # [B, T]
        old_logprobs = rollout.student_logprobs.detach()         # [B, T]
        teacher_rewards = rollout.teacher_rewards.detach()       # [B, T]

        current_logits = self._compute_response_logits(
            model=student,
            full_ids=rollout.full_ids,
            full_attention_mask=rollout.full_attention_mask,
            response_ids=rollout.response_ids,
        )
        current_logprobs = self._compute_token_logprobs(current_logits, rollout.response_ids)

        logprob_diff = ((current_logprobs - old_logprobs).abs() * mask).sum() / mask.sum().clamp_min(1.0)

        # reviewed-style reward construction
        old_rewards = teacher_rewards - old_logprobs
        old_rewards = old_rewards * mask

        if self.cfg.cliprange_reward is not None:
            old_rewards = torch.clamp(
                old_rewards,
                -self.cfg.cliprange_reward,
                self.cfg.cliprange_reward,
            )

        response_length = rollout.response_ids.size(1)
        advantages = self._get_advantages_and_returns(
            rewards=old_rewards,
            response_length=response_length,
            mask=mask,
            use_whitening=self.cfg.whiten_advantages,
        )

        w = rollout.w.detach()

        pg_loss = self._pg_loss(
            logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            w=w,
        )

        reg_loss = self._reg_loss(
            rollout=rollout,
            student_logits=current_logits,
            mask=mask,
        )

        loss = pg_loss
        if self.cfg.single_step_reg:
            loss = loss + self.cfg.single_step_reg_coef * reg_loss

        cumsum_rewards = self._get_cumsum_rewards(old_rewards)
        cumsum_rewards = cumsum_rewards.mean().item()

        stats = {
            "rl_loss": float(loss.detach().item()),
            "pg_loss": float(pg_loss.detach().item()),
            "reg_loss": float(reg_loss.detach().item()),
            "reward_mean": float((old_rewards * mask).sum().detach().item() / mask.sum().clamp_min(1.0).item()),
            "cumsum_reward": float(cumsum_rewards),
            "old_logprob_mean": float((old_logprobs * mask).sum().detach().item() / mask.sum().clamp_min(1.0).item()),
            "cur_logprob_mean": float((current_logprobs * mask).sum().detach().item() / mask.sum().clamp_min(1.0).item()),
            "logprob_diff": float(logprob_diff.detach().item()),
            "response_len": float(mask.sum(dim=-1).float().mean().detach().item()),
        }

        return loss, stats

    # ------------------------------------------------------------
    # Reviewed-style PT loss adapted to your code
    # ------------------------------------------------------------
    def pt_loss(self, student, batch):
        stats = {}
        model_batch, no_model_batch = batch

        outputs = student(
            **model_batch,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits

        lm_loss = self.ce_loss(
            logits.view(-1, logits.size(-1)),
            no_model_batch["label"].view(-1)
        )

        kd_loss = torch.tensor(0.0, device=logits.device)
        loss = lm_loss

        if self.teacher is not None and self.cfg.kd_ratio is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    **model_batch,
                    return_dict=True,
                    use_cache=False,
                )
                teacher_logits = teacher_outputs.logits

            valid_mask = (no_model_batch["label"] != -100).float()

            teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
            student_log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

            kd_per_token = -(teacher_probs * student_log_probs).sum(dim=-1)
            kd_loss = (kd_per_token * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

            loss = (1 - self.cfg.kd_ratio) * lm_loss + self.cfg.kd_ratio * kd_loss

        stats["pt_loss"] = float(loss.detach().item())
        stats["lm_loss"] = float(lm_loss.detach().item())
        stats["kd_loss"] = float(kd_loss.detach().item())

        return loss, stats
# ============================================================
# 6. Model loading
# ============================================================

def load_models(cfg: Config, device: torch.device):
    print(f"\nLoading teacher from: {cfg.teacher_model_name}")
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_model_name)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Loading student from: {cfg.student_init_path}")
    student = PPOModel(cfg.student_init_path, device)

    return teacher, student



class MiniLLMTrainer:
    def __init__(
        self,
        cfg,
        tokenizer,
        student,
        teacher,
        reward,
        loss_fn,
        train_prompt_loader,
        valid_prompt_loader,
        train_lm_loader=None,
        valid_lm_loader=None,
        device=None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.student = student
        self.teacher = teacher
        self.reward = reward
        self.loss_fn = loss_fn

        self.train_prompt_loader = train_prompt_loader
        self.valid_prompt_loader = valid_prompt_loader
        self.train_lm_loader = train_lm_loader
        self.valid_lm_loader = valid_lm_loader

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = AdamW(
            self.student.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.global_step = 0
        self.best_rouge = -1.0
        self.sampler = MiniSampler(
        student=self.student,
        teacher=self.teacher,
        reward=self.reward,
        prompt_loader=self.train_prompt_loader,
        tokenizer=self.tokenizer,
        cfg=self.cfg,
        device=self.device,
        )

        self.store = MiniRolloutStore()

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        os.makedirs(self.cfg.best_model_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def move_batch_to_device(self, batch):
        model_batch, no_model_batch = batch

        for k in model_batch:
            model_batch[k] = model_batch[k].to(self.device)

        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(self.device)

        return model_batch, no_model_batch

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        # student is PPOModel wrapper
        self.student.base_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        print(f"Model saved to: {save_dir}")

    @torch.no_grad()
    def generate_response_texts(self, prompt_loader, max_new_tokens=None, max_batches=None):
        self.student.eval()

        all_pred_texts = []
        all_ref_texts = []

        for step, batch in enumerate(prompt_loader):
            model_batch, no_model_batch = batch

            input_ids = model_batch["input_ids"].to(self.device)
            attention_mask = model_batch["attention_mask"].to(self.device)

            outputs = self.student.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or self.cfg.eval_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            prompt_len = input_ids.size(1)
            response_ids = outputs[:, prompt_len:]

            pred_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            all_pred_texts.extend(pred_texts)

            # reference texts are not in model_batch anymore, so get them from dataset batch if needed
            # If your prompt collate still returns them elsewhere, adjust accordingly.
            # Here we infer from the original batch structure if available.
            if isinstance(batch, tuple) and len(batch) == 2:
                # no direct text refs inside tensors, so skip unless separately passed
                pass

            if max_batches is not None and (step + 1) >= max_batches:
                break

        return all_pred_texts, all_ref_texts


    @torch.no_grad()
    def evaluate_pt(self):
        if self.valid_lm_loader is None:
            return {}

        self.student.eval()

        total_pt = 0.0
        total_lm = 0.0
        total_kd = 0.0
        count = 0

        for batch in self.valid_lm_loader:
            lm_model_batch, lm_no_model_batch = self.move_batch_to_device(batch)
            pt_loss, pt_stats = self.loss_fn.pt_loss(
                student=self.student,
                batch=(lm_model_batch, lm_no_model_batch),
            )

            total_pt += pt_stats["pt_loss"]
            total_lm += pt_stats["lm_loss"]
            total_kd += pt_stats["kd_loss"]
            count += 1

        return {
            "pt_loss": total_pt / max(count, 1),
            "lm_loss": total_lm / max(count, 1),
            "kd_loss": total_kd / max(count, 1),
        }

    @torch.no_grad()
    def evaluate_ppo(self, prompt_loader=None, max_batches=None):
        from math import isnan

        prompt_loader = prompt_loader or self.valid_prompt_loader
        self.student.eval()

        total_rl = 0.0
        total_pg = 0.0
        total_reg = 0.0
        total_reward = 0.0
        total_resp_len = 0.0
        count = 0

        for step, batch in enumerate(prompt_loader):
            model_batch, _ = batch

            rollout = collect_rollout_batch(
                student=self.student,
                teacher=self.teacher,
                reward=self.reward,
                model_batch=model_batch,
                tokenizer=self.tokenizer,
                cfg=self.cfg,
                device=self.device,
            )

            rl_loss, rl_stats = self.loss_fn.rollout_loss(
                student=self.student,
                rollout=rollout,
            )

            total_rl += rl_stats["rl_loss"]
            total_pg += rl_stats["pg_loss"]
            total_reg += rl_stats["reg_loss"]
            total_reward += rl_stats["reward_mean"]
            total_resp_len += rl_stats["response_len"]
            count += 1

            if max_batches is not None and (step + 1) >= max_batches:
                break

        return {
            "rl_loss": total_rl / max(count, 1),
            "pg_loss": total_pg / max(count, 1),
            "reg_loss": total_reg / max(count, 1),
            "reward_mean": total_reward / max(count, 1),
            "response_len": total_resp_len / max(count, 1),
        }

    
    @torch.no_grad()
    def evaluate_rouge_l(self, prompt_loader, max_batches=None):
        """
        Compute paper-style Rouge-L / Exact Match using compute_metrics().
        Assumes prompt_loader.dataset has prompt_text / reference_text examples.
        """
        self.student.eval()

        predictions = []
        references = []
        seen = 0

        for step, batch in enumerate(prompt_loader):
            model_batch, _ = batch

            input_ids = model_batch["input_ids"].to(self.device)
            attention_mask = model_batch["attention_mask"].to(self.device)

            outputs = self.student.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.cfg.eval_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            prompt_len = input_ids.size(1)
            response_ids = outputs[:, prompt_len:]
            pred_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            batch_size = input_ids.size(0)
            start_idx = step * batch_size
            end_idx = start_idx + batch_size

            # compute_metrics expects each reference to be a list[str]
            refs = [
                [prompt_loader.dataset[i]["reference_text"]]
                for i in range(start_idx, min(end_idx, len(prompt_loader.dataset)))
            ]

            for pred, ref in zip(pred_responses, refs):
                predictions.append(pred)
                references.append(ref)
                seen += 1

                if self.cfg.num_eval_samples is not None and seen >= self.cfg.num_eval_samples:
                    return compute_metrics(predictions, references, xlingual=False)

            if max_batches is not None and (step + 1) >= max_batches:
                break

        return compute_metrics(predictions, references, xlingual=False)

    def evaluate(self):
        eval_rl = self.evaluate_ppo()
        eval_pt = self.evaluate_pt() if self.valid_lm_loader is not None else {}
        text_metrics = self.evaluate_rouge_l(self.valid_prompt_loader)

        results = {}
        results.update(eval_rl)
        results.update(eval_pt)
        results.update(text_metrics)

        log_str = "eval | " + " | ".join([f"{k}: {v:.4f}" for k, v in results.items()])
        print(log_str)

        return results
    
    @torch.no_grad()
    def test_student_generation(self, prompt, max_new_tokens=100):
        self.student.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.student.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(text)
        return text

    # ------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------
    def train(self):
        lm_iterator = cycle(self.train_lm_loader) if self.train_lm_loader is not None else None

        self.optimizer.zero_grad(set_to_none=True)
        update_step = 0

        while update_step < self.cfg.max_steps:
            # ----------------------------------------------------
            # 1) Collect fresh rollouts FIRST
            # ----------------------------------------------------
            self.store.clear()
            new_rollouts = self.sampler.sample_many(self.cfg.num_rollout_batches)
            self.store.push(new_rollouts)

            # ----------------------------------------------------
            # 2) Train on stored rollouts
            # ----------------------------------------------------
            for ppo_epoch in range(self.cfg.ppo_epochs):
                for rollout in self.store.history:
                    # keep dropout off for RL-style update
                    self.student.eval()

                    rl_loss, rl_stats = self.loss_fn.rollout_loss(
                        student=self.student,
                        rollout=rollout,
                    )

                    # optional LM + KD branch
                    if lm_iterator is not None:
                        lm_batch = next(lm_iterator)
                        lm_model_batch, lm_no_model_batch = self.move_batch_to_device(lm_batch)

                        pt_loss, pt_stats = self.loss_fn.pt_loss(
                            student=self.student,
                            batch=(lm_model_batch, lm_no_model_batch),
                        )
                    else:
                        pt_loss = torch.tensor(0.0, device=self.device)
                        pt_stats = {
                            "pt_loss": 0.0,
                            "lm_loss": 0.0,
                            "kd_loss": 0.0,
                        }

                    total_loss = rl_loss + self.cfg.lm_coef * pt_loss
                    (total_loss / self.cfg.gradient_accumulation_steps).backward()

                    self.global_step += 1

                    if self.global_step % self.cfg.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        update_step += 1

                        if update_step % self.cfg.log_every == 0:
                            print(
                                f"Update {update_step} | "
                                f"total={total_loss.item():.6f} | "
                                f"rl={rl_stats['rl_loss']:.6f} | "
                                f"pg={rl_stats['pg_loss']:.6f} | "
                                f"reg={rl_stats['reg_loss']:.6f} | "
                                f"logprob_diff={rl_stats['logprob_diff']:.6f} | "
                                f"pt={pt_stats['pt_loss']:.6f} | "
                                f"lm={pt_stats['lm_loss']:.6f} | "
                                f"kd={pt_stats['kd_loss']:.6f} | "
                                f"reward={rl_stats['reward_mean']:.6f} | "
                                f"resp_len={rl_stats['response_len']:.2f}"
                                f"cumsum_reward={rl_stats['cumsum_reward']:.6f} | "
                            )

                        if update_step > 0 and update_step % self.cfg.eval_every == 0:
                            results = self.evaluate()

                            if results["rougeL"] > self.best_rouge:
                                self.best_rouge = results["rougeL"]
                                self.save_model(self.cfg.best_model_dir)
                                print(f"New best model saved to: {self.cfg.best_model_dir}")
                                print(f"Updated best Rouge-L: {self.best_rouge:.6f}")

                        if update_step > 0 and update_step % self.cfg.save_every == 0:
                            save_dir = os.path.join(self.cfg.output_dir, f"step_{update_step}")
                            self.save_model(save_dir)

                        if update_step >= self.cfg.max_steps:
                            print(f"\nTraining complete. Best validation Rouge-L: {self.best_rouge:.6f}")
                            return



# ============================================================
# 13. Main
# ============================================================

def main():
    cfg = CFG

    # 1. setup
    set_seed(cfg.seed)
    device = get_device()

    # 2. data
    ds = load_dolly_dataset(cfg.train_file, cfg.valid_file)
    ds = preprocess_dataset(ds)

    # 3. tokenizer
    tokenizer = load_tokenizer(cfg.student_init_path)

    # 4. dataloaders
    train_prompt_loader = build_prompt_dataloader(
        dataset_split=ds["train"],
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_total_length,
        shuffle=True,
    )

    valid_prompt_loader = build_prompt_dataloader(
        dataset_split=ds["validation"],
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_total_length,
        shuffle=False,
    )
    sample = valid_prompt_loader.dataset[0]
    print(sample.keys())
    print(sample["prompt_text"][:200])
    print(sample["reference_text"][:200])
    train_lm_loader = build_lm_dataloader(
    dataset_split=ds["train"],
    tokenizer=tokenizer,
    batch_size=cfg.batch_size,
    max_length=cfg.max_total_length,
    model_type="gpt2",
    shuffle=True,
)

    valid_lm_loader = build_lm_dataloader(
        dataset_split=ds["validation"],
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_total_length,
        model_type="gpt2",
        shuffle=False,
    )

    # 5. models
    teacher, student = load_models(cfg, device)

    # 6. reward + loss
    reward = Reward(cfg, tokenizer, teacher)
    loss_fn = MiniLLMLoss(cfg, tokenizer, teacher)

    # 7. trainer
    trainer = MiniLLMTrainer(
        cfg=cfg,
        tokenizer=tokenizer,
        student=student,
        teacher=teacher,
        reward=reward,
        loss_fn=loss_fn,
        train_prompt_loader=train_prompt_loader,
        valid_prompt_loader=valid_prompt_loader,
        train_lm_loader=train_lm_loader,
        valid_lm_loader=valid_lm_loader,
        device=device,
    )

    # 8. train
    trainer.train()

    # 9. optional test generation
    print("\n--- Generation from best model ---")
    trainer.test_student_generation(
    prompt="""### Instruction:
Explain what diffusion MRI measures.

### Response:
""",
    max_new_tokens=100,
)


if __name__ == "__main__":
    main()