#!/usr/bin/env python
"""Run pre-scale study gates for continuous embedding training.

This script implements the following diagnostics before large-scale training:
  - Gate A-Overfit: embedding/readout roundtrip sanity on fixed tiny batches.
  - Gate A-Recon: forward-noise reconstruction sanity at fixed timesteps.
  - Gate B: JVP + multi-rank consistency check on the intended attention backend.
  - Gate C: Infill clamp invariance (token-id and embedding drift checks).
  - Gate D-Stability / D-Signal: deterministic internal metrics and early learning signal.
  - Gate E / E2: throughput/memory profiling and objective-comparison budget check.
  - Gate F: CE gradient-path sanity (denoiser/decoder/embedding grad norms).
  - Gate G: mask semantics checks for all-zero/all-one span masks.

All artifacts are written to a single timestamped folder for external review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import socket
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from discrete_diffusion.algorithms.continuous_embedding_diffusion import (
    ContinuousEmbeddingDiffusion,
)
from discrete_diffusion.continuous.diagnostics.jvp_probe import probe_model_jvp
from discrete_diffusion.evaluations.continuous_capabilities.sampling import (
    decode_embeddings,
    sample_latents_conditioned,
)
from discrete_diffusion.models.common import set_sdpa_math_mode, supports_flash_attention
from discrete_diffusion.sampling.continuous_embedding import ContinuousEmbeddingSampler


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, L]
    attention_mask: torch.Tensor  # [B, L]
    span_mask: torch.Tensor  # [B, L], 1=generate, 0=condition


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _stage(msg: str) -> None:
    print(f"[gates] {msg}", flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        yield
        return
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    if device.type == "cpu":
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            yield
        return
    with nullcontext():
        yield


def _safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _ensure_jvp_safe_backend(algo: ContinuousEmbeddingDiffusion) -> dict[str, Any]:
    """Probe JVP support and switch to SDPA math kernels if needed."""
    if getattr(algo, "objective_name", "") not in {"meanflow", "imf"}:
        return {
            "objective": str(getattr(algo, "objective_name", "")),
            "initial_ok": True,
            "fallback_applied": False,
            "final_ok": True,
            "message": "not_applicable",
        }

    set_sdpa_math_mode(False)
    initial = probe_model_jvp(algo)
    if initial.ok:
        return {
            "objective": str(algo.objective_name),
            "initial_ok": True,
            "fallback_applied": False,
            "final_ok": True,
            "message": initial.message,
        }

    if hasattr(algo.denoiser, "set_attention_backend"):
        algo.denoiser.set_attention_backend("sdpa")
    set_sdpa_math_mode(True)
    final = probe_model_jvp(algo)
    return {
        "objective": str(algo.objective_name),
        "initial_ok": bool(initial.ok),
        "fallback_applied": True,
        "final_ok": bool(final.ok),
        "message": final.message,
        "initial_message": initial.message,
    }


def _printable_ratio(texts: list[str]) -> float:
    vals = []
    for text in texts:
        if not text:
            vals.append(0.0)
            continue
        printable = sum(1 for ch in text if 32 <= ord(ch) <= 126)
        vals.append(printable / float(len(text)))
    return float(np.mean(vals)) if vals else 0.0


def _build_cfg(
    *,
    seq_len: int,
    embed_dim: int,
    hidden_size: int,
    n_heads: int,
    n_blocks: int,
    objective: str,
    interpolant: str,
    attn_backend: str,
    gradient_checkpointing: bool,
    embedding_provider: str,
    embedding_trainable: bool,
    lambda_ce: float,
    self_conditioning: bool,
    self_conditioning_prob: float,
    min_snr_gamma: Optional[float],
    imf_alpha: float,
    imf_warmup_steps: int,
    imf_ramp_steps: int,
    train_timesteps: int,
    vocab_size: int,
) -> OmegaConf:
    cfg = {
        "algo": {
            "_target_": "discrete_diffusion.algorithms.continuous_embedding_diffusion.ContinuousEmbeddingDiffusion",
            "name": "continuous_embedding_diffusion",
            "stage": 1,
            "sampling_eps": 1e-3,
            "loss_type": "simple",
            "parameterization": "x0",
            "objective": objective,
            "interpolant": interpolant,
            "conditioning": "span_masking",
            "embedding_provider": embedding_provider,
            "embedding_trainable": embedding_trainable,
            "train_timesteps": train_timesteps,
            "lambda_ce": float(lambda_ce),
            "stabilizers": {
                "self_conditioning": bool(self_conditioning),
                "self_conditioning_prob": float(self_conditioning_prob),
                "anchor_weight": 0.0,
                "noise_scale": 1.0,
            },
            "loss": {
                "min_snr_gamma": min_snr_gamma,
            },
            "imf": {
                "dt_min": 1e-3,
                "stopgrad_dudt": True,
                "use_aux_v_head": True,
                "alpha_floor": 0.0,
                "alpha_imf": float(imf_alpha),
                "warmup_steps": int(imf_warmup_steps),
                "ramp_steps": int(imf_ramp_steps),
                "fm_aux_weight": 0.05,
            },
            "span_min_spans": 1,
            "span_max_spans": 3,
            "span_length_dist": "geometric",
            "span_mean_length": 3.0,
            "decoder": {"stage2": {"enabled": False}},
            "sampler": {
                "_target_": "discrete_diffusion.sampling.continuous_embedding.ContinuousEmbeddingSampler",
                "sampling_method": "rectified_one_step",
                "eta": 0.0,
                "temperature": 1.0,
                "guidance_scale": 1.0,
                "clip_denoised": False,
            },
        },
        "model": {
            "embed_dim": int(embed_dim),
            "hidden_size": int(hidden_size),
            "n_heads": int(n_heads),
            "n_blocks": int(n_blocks),
            "mlp_ratio": 2,
            "dropout": 0.1,
            "time_embed_dim": int(hidden_size),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "decoder_hidden_size": int(hidden_size),
            "decoder_n_heads": int(n_heads),
            "decoder_n_blocks": 1,
            "length": int(seq_len),
            "attn_backend": attn_backend,
        },
        "noise": {
            "_target_": "discrete_diffusion.noise_schedules.LinearNoiseSchedule",
            "eps": 1e-4,
        },
        "forward_process": {
            "_target_": "discrete_diffusion.forward_process.gaussian.GaussianForwardProcess",
            "clip_noise": False,
            "noise_clip_value": 6.0,
            "per_token_noise": True,
        },
        "training": {
            "ema": 0.0,
            "antithetic_sampling": True,
            "loss_precision": "bf16",
        },
        "optim": {
            "lr": 3e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "lr_scheduler": {
            "_target_": "torch.optim.lr_scheduler.ConstantLR",
            "factor": 1.0,
            "total_iters": 1,
        },
        "vocab_size": int(vocab_size),
    }
    return OmegaConf.create(cfg)


def _ensure_special_tokens(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "mask_token_id", None) is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})


def _load_owt_tensors(
    *,
    tokenizer,
    seq_len: int,
    num_examples: int,
    cache_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Prefer streaming to avoid full corpus materialization for small gate studies.
    try:
        ds = load_dataset(
            "openwebtext",
            split="train",
            cache_dir=cache_dir,
            streaming=True,
        )
        texts = []
        for i, row in enumerate(ds):
            texts.append(str(row["text"]))
            if i + 1 >= int(num_examples):
                break
    except Exception:
        # Fallback for environments where streaming is unavailable.
        ds = load_dataset(
            "openwebtext",
            split=f"train[:{int(num_examples)}]",
            cache_dir=cache_dir,
        )
        texts = [str(x) for x in ds["text"]]
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=int(seq_len),
    )
    ids = tok["input_ids"]
    mask = tok["attention_mask"]
    # Keep only non-trivial rows.
    valid = mask.sum(dim=1) >= 8
    if int(valid.sum().item()) == 0:
        raise RuntimeError("No usable OWT rows found after tokenization.")
    return ids[valid], mask[valid]


def _load_synthetic_tensors(
    *,
    seq_len: int,
    num_examples: int,
    vocab_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    ids = torch.randint(
        low=0,
        high=max(int(vocab_size), 2),
        size=(int(num_examples), int(seq_len)),
        generator=gen,
        dtype=torch.long,
    )
    attn = torch.ones_like(ids)
    return ids, attn


def _deterministic_span_mask(
    attention_mask: torch.Tensor,
    *,
    pattern_id: int,
    num_patterns: int,
) -> torch.Tensor:
    """Build deterministic span masks: 1=generate, 0=condition."""
    bsz, seqlen = attention_mask.shape
    out = torch.zeros_like(attention_mask)
    for b in range(bsz):
        valid_len = int(attention_mask[b].sum().item())
        if valid_len <= 4:
            out[b, 1:2] = 1
            continue
        frac_start = (pattern_id + 1) / float(num_patterns + 2)
        frac_len = 0.18 + 0.02 * (pattern_id % 3)
        start = max(1, int(valid_len * frac_start))
        span_len = max(1, int(valid_len * frac_len))
        end = min(valid_len - 1, start + span_len)
        if end <= start:
            end = min(valid_len, start + 1)
        out[b, start:end] = 1
        if int(out[b].sum().item()) == 0:
            out[b, 1:2] = 1
    return out


def _to_batches(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
    num_batches: int,
    num_patterns: int,
) -> list[Batch]:
    needed = int(batch_size) * int(num_batches)
    if input_ids.shape[0] < needed:
        reps = int(math.ceil(needed / float(input_ids.shape[0])))
        input_ids = input_ids.repeat((reps, 1))
        attention_mask = attention_mask.repeat((reps, 1))
    batches: list[Batch] = []
    for i in range(num_batches):
        lo = i * batch_size
        hi = (i + 1) * batch_size
        ids = input_ids[lo:hi].clone()
        attn = attention_mask[lo:hi].clone()
        span = _deterministic_span_mask(attn, pattern_id=i % max(num_patterns, 1), num_patterns=max(num_patterns, 1))
        batches.append(Batch(input_ids=ids, attention_mask=attn, span_mask=span))
    return batches


def _masked_accuracy(logits: torch.Tensor, target_ids: torch.Tensor, token_mask: torch.Tensor, attention_mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    valid = token_mask.bool() & attention_mask.bool()
    n = int(valid.sum().item())
    if n == 0:
        return 0.0
    correct = int((pred[valid] == target_ids[valid]).sum().item())
    return float(correct / n)


def _compute_stage1_losses(
    algo: ContinuousEmbeddingDiffusion,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    span_mask: Optional[torch.Tensor],
    ce_token_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    z0 = algo._embed_inputs(input_ids, attention_mask)
    if algo.objective_name in {"meanflow", "imf"} and hasattr(algo.objective, "config"):
        algo.objective.config.alpha_imf = float(algo._current_imf_alpha())

    out = algo.objective.compute(
        model=algo,
        x0=z0,
        attention_mask=attention_mask,
        span_mask=span_mask,
        input_ids=input_ids,
    )
    x0_hat = out.x0_hat if out.x0_hat is not None else z0
    ce = algo._compute_ce_loss(
        x0_hat,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_mask=ce_token_mask,
    )
    total = out.loss + algo.lambda_ce * ce
    logits = algo._decode_logits(x0_hat)
    token_mask_for_acc = ce_token_mask if ce_token_mask is not None else attention_mask
    stats = {
        "main_loss": float(out.loss.detach().cpu().item()),
        "ce_loss": float(ce.detach().cpu().item()),
        "masked_acc": _masked_accuracy(logits.detach(), input_ids, token_mask_for_acc, attention_mask),
        "x0_hat_source": "objective" if out.x0_hat is not None else "fallback_z0",
    }
    return total, stats, x0_hat


def _compute_fixed_stage1_loss(
    algo: ContinuousEmbeddingDiffusion,
    batch: Batch,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    return _compute_stage1_losses(
        algo,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        span_mask=batch.span_mask,
        ce_token_mask=batch.span_mask,
    )


def _predict_rectified_direction(
    model: ContinuousEmbeddingDiffusion,
    z_t: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    sampler = ContinuousEmbeddingSampler(config=model.config, sampling_method="rectified_one_step", eta=0.0)
    return sampler._predict_rectified_direction(model, z_t, t, r=r)


@torch.no_grad()
def _sample_rectified_conditioned(
    model: ContinuousEmbeddingDiffusion,
    *,
    conditioning_ids: torch.Tensor,
    fixed_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    num_steps: int,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = conditioning_ids.device
    z_known = model._embed_inputs(conditioning_ids, attention_mask)
    noise = torch.randn(
        z_known.shape,
        device=device,
        dtype=z_known.dtype,
        generator=generator,
    )
    m = fixed_mask.bool().unsqueeze(-1)
    z_t = torch.where(m, z_known, noise)

    if num_steps <= 1:
        t = torch.ones(z_t.shape[0], device=device, dtype=z_t.dtype)
        r = torch.zeros(z_t.shape[0], device=device, dtype=z_t.dtype)
        direction = _predict_rectified_direction(model, z_t, t, r)
        z_t = z_t - direction
        z_t = torch.where(m, z_known, z_t)
    else:
        times = torch.linspace(1.0, 0.0, int(num_steps) + 1, device=device, dtype=z_t.dtype)
        for i in range(int(num_steps)):
            t = times[i].expand(z_t.shape[0])
            r = times[i + 1].expand(z_t.shape[0])
            direction = _predict_rectified_direction(model, z_t, t, r)
            dt = (t - r).view(-1, 1, 1)
            z_t = z_t - dt * direction
            z_t = torch.where(m, z_known, z_t)

    logits = model._decode_logits(z_t)
    token_ids = logits.argmax(dim=-1)
    return z_t, token_ids


def _sample_conditioned_embeddings(
    model: ContinuousEmbeddingDiffusion,
    *,
    conditioning_ids: torch.Tensor,
    span_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    num_steps: int,
    sampling_method: str,
    eta: float,
    eps: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    fixed_mask = (~span_mask.bool()).to(dtype=torch.bool)
    if model.interpolant_name == "rectified":
        z, ids = _sample_rectified_conditioned(
            model,
            conditioning_ids=conditioning_ids,
            fixed_mask=fixed_mask,
            attention_mask=attention_mask,
            num_steps=num_steps,
            generator=generator,
        )
        # Infill contract: conditioned tokens must remain unchanged.
        ids = torch.where(span_mask.bool(), ids, conditioning_ids)
        return z, ids

    # VP fallback for DDPM-style models.
    z = sample_latents_conditioned(
        model,
        conditioning_ids=conditioning_ids,
        fixed_mask=fixed_mask,
        attention_mask=attention_mask,
        num_steps=num_steps,
        sampling_method=sampling_method,
        eta=eta,
        eps=eps,
        clip_denoised=False,
        clip_range=(-10.0, 10.0),
    )
    ids, _ = decode_embeddings(
        model,
        tokenizer=None,  # not needed for token-ID output
        embeddings=z,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        ban_special_tokens=False,
    )
    ids = torch.where(span_mask.bool(), ids, conditioning_ids)
    return z, ids


def _build_conditioning_ids(tokenizer, batch: Batch) -> torch.Tensor:
    ids = batch.input_ids.clone()
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is not None:
        ids[batch.span_mask.bool()] = int(mask_token_id)
    return ids


def _evaluate_batches(
    algo: ContinuousEmbeddingDiffusion,
    batches: list[Batch],
) -> dict[str, float]:
    algo.eval()
    model_device = next(algo.parameters()).device
    losses = []
    ces = []
    accs = []
    with torch.no_grad():
        for batch in batches:
            b = Batch(
                input_ids=batch.input_ids.to(model_device),
                attention_mask=batch.attention_mask.to(model_device),
                span_mask=batch.span_mask.to(model_device),
            )
            total, stats, _ = _compute_fixed_stage1_loss(algo, b)
            losses.append(float(total.detach().cpu().item()))
            ces.append(float(stats["ce_loss"]))
            accs.append(float(stats["masked_acc"]))
    return {
        "loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "ce_mean": float(np.mean(ces)) if ces else float("nan"),
        "masked_acc_mean": float(np.mean(accs)) if accs else float("nan"),
    }


def _broadcast_time(t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = t
    while out.ndim < target.ndim:
        out = out.unsqueeze(-1)
    return out


@torch.no_grad()
def _predict_x0_from_forward_noised(
    algo: ContinuousEmbeddingDiffusion,
    *,
    z_t: torch.Tensor,
    x0: torch.Tensor,
    t: torch.Tensor,
    attention_mask: torch.Tensor,
    span_mask: torch.Tensor,
) -> torch.Tensor:
    if algo.interpolant_name == "rectified":
        if algo.objective_name == "flow_matching":
            direction = algo.predict_velocity(
                z_t,
                t,
                attention_mask=attention_mask,
                span_mask=span_mask,
            )
        else:
            r = torch.zeros_like(t)
            direction = algo.predict_u(
                z_t,
                r,
                t,
                attention_mask=attention_mask,
                span_mask=span_mask,
            )
        x0_hat = z_t - _broadcast_time(t, z_t) * direction
    else:
        alpha_t = algo.noise.alpha_t(t)
        alpha_t = _broadcast_time(alpha_t, z_t)
        x0_hat = algo.denoiser.predict_x0(z_t, t, alpha_t, attention_mask)

    m = span_mask.bool()
    while m.ndim < x0.ndim:
        m = m.unsqueeze(-1)
    return torch.where(m, x0_hat, x0)


@torch.no_grad()
def _evaluate_reconstruction_metrics(
    algo: ContinuousEmbeddingDiffusion,
    batches: list[Batch],
    *,
    device: torch.device,
    t_values: list[float],
    seed: int,
) -> dict[str, Any]:
    algo.eval()
    per_t: list[dict[str, Any]] = []
    runtime_errors: list[str] = []

    for t_val in t_values:
        ce_vals = []
        acc_vals = []
        for i, b0 in enumerate(batches):
            try:
                batch = Batch(
                    input_ids=b0.input_ids.to(device),
                    attention_mask=b0.attention_mask.to(device),
                    span_mask=b0.span_mask.to(device),
                )
                x0 = algo._embed_inputs(batch.input_ids, batch.attention_mask)
                gen = torch.Generator(device=device).manual_seed(int(seed + 1000 * round(t_val, 4) + i))
                noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=gen)

                t = torch.full((x0.shape[0],), float(t_val), device=device, dtype=x0.dtype)
                m = batch.span_mask.bool()
                while m.ndim < x0.ndim:
                    m = m.unsqueeze(-1)

                if algo.interpolant_name == "rectified":
                    z_masked = (1.0 - _broadcast_time(t, x0)) * x0 + _broadcast_time(t, x0) * noise
                else:
                    alpha_t = algo.noise.alpha_t(t).to(dtype=x0.dtype)
                    alpha_t = _broadcast_time(alpha_t, x0)
                    z_masked = torch.sqrt(alpha_t.clamp(min=1e-20)) * x0 + torch.sqrt((1.0 - alpha_t).clamp(min=1e-20)) * noise
                z_t = torch.where(m, z_masked, x0)

                x0_hat = _predict_x0_from_forward_noised(
                    algo,
                    z_t=z_t,
                    x0=x0,
                    t=t,
                    attention_mask=batch.attention_mask,
                    span_mask=batch.span_mask,
                )
                logits = algo._decode_logits(x0_hat)
                ce = algo._compute_ce_loss(
                    x0_hat,
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    token_mask=batch.span_mask,
                )
                acc = _masked_accuracy(logits, batch.input_ids, batch.span_mask, batch.attention_mask)
                ce_vals.append(float(ce.detach().cpu().item()))
                acc_vals.append(float(acc))
            except RuntimeError as exc:
                runtime_errors.append(f"t={t_val:.4f}, batch={i}: {exc}")

        per_t.append(
            {
                "t": float(t_val),
                "masked_ce_mean": float(np.mean(ce_vals)) if ce_vals else float("nan"),
                "masked_acc_mean": float(np.mean(acc_vals)) if acc_vals else float("nan"),
                "num_batches": int(len(ce_vals)),
            }
        )

    return {
        "per_t": per_t,
        "runtime_errors": runtime_errors,
    }


@torch.no_grad()
def _evaluate_roundtrip_metrics(
    algo: ContinuousEmbeddingDiffusion,
    batches: list[Batch],
    *,
    device: torch.device,
) -> dict[str, float]:
    algo.eval()
    ce_vals = []
    acc_vals = []
    for b0 in batches:
        batch = Batch(
            input_ids=b0.input_ids.to(device),
            attention_mask=b0.attention_mask.to(device),
            span_mask=b0.span_mask.to(device),
        )
        x0 = algo._embed_inputs(batch.input_ids, batch.attention_mask)
        logits = algo._decode_logits(x0)
        ce = algo._compute_ce_loss(
            x0,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            token_mask=batch.attention_mask,
        )
        acc = _masked_accuracy(logits, batch.input_ids, batch.attention_mask, batch.attention_mask)
        ce_vals.append(float(ce.detach().cpu().item()))
        acc_vals.append(float(acc))
    return {
        "roundtrip_ce_mean": float(np.mean(ce_vals)) if ce_vals else float("nan"),
        "roundtrip_acc_mean": float(np.mean(acc_vals)) if acc_vals else float("nan"),
    }


def _roundtrip_trainable_parameters(algo: ContinuousEmbeddingDiffusion) -> list[torch.nn.Parameter]:
    provider_name = str(getattr(algo, "embedding_provider_name", "legacy_contextual")).lower()
    if provider_name in {"lookup", "tied"} and isinstance(algo.embedding_provider, torch.nn.Module):
        return [p for p in algo.embedding_provider.parameters() if p.requires_grad]
    return [p for p in algo.decoder.parameters() if p.requires_grad]


def _run_roundtrip_fit(
    algo: ContinuousEmbeddingDiffusion,
    *,
    overfit_batches: list[Batch],
    steps: int,
    lr: float,
    grad_clip: float,
    autocast_bf16: bool,
    device: torch.device,
) -> dict[str, Any]:
    params = _roundtrip_trainable_parameters(algo)
    if not params:
        return {
            "performed": False,
            "reason": "no_trainable_roundtrip_parameters",
            "curve": {"step": [], "ce": [], "acc": []},
            "post": _evaluate_roundtrip_metrics(algo, overfit_batches, device=device),
        }

    algo.train()
    opt = torch.optim.AdamW(params, lr=lr)
    curve = {"step": [], "ce": [], "acc": []}
    runtime_errors: list[str] = []
    nan_failures = 0

    for step in range(int(steps)):
        batch = overfit_batches[step % len(overfit_batches)]
        b = Batch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            span_mask=batch.span_mask.to(device),
        )
        opt.zero_grad(set_to_none=True)
        try:
            with _autocast_context(device, enabled=autocast_bf16):
                x0 = algo._embed_inputs(b.input_ids, b.attention_mask)
                logits = algo._decode_logits(x0)
                loss = algo._compute_ce_loss(
                    x0,
                    input_ids=b.input_ids,
                    attention_mask=b.attention_mask,
                    token_mask=b.attention_mask,
                )
                acc = _masked_accuracy(logits, b.input_ids, b.attention_mask, b.attention_mask)
        except RuntimeError as exc:
            runtime_errors.append(str(exc))
            break

        if not torch.isfinite(loss):
            nan_failures += 1
            runtime_errors.append(f"non-finite roundtrip-fit loss at step={step}")
            break

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        opt.step()

        curve["step"].append(step + 1)
        curve["ce"].append(float(loss.detach().cpu().item()))
        curve["acc"].append(float(acc))

    post = _evaluate_roundtrip_metrics(algo, overfit_batches, device=device)
    return {
        "performed": True,
        "runtime_errors": runtime_errors,
        "nan_failures": int(nan_failures),
        "curve": curve,
        "post": post,
    }


def _lookup_t_metric(recon: dict[str, Any], t: float) -> dict[str, float]:
    entries = recon.get("per_t", [])
    if not entries:
        return {"t": t, "masked_ce_mean": float("nan"), "masked_acc_mean": float("nan")}
    best = min(entries, key=lambda x: abs(float(x["t"]) - float(t)))
    return {
        "t": float(best["t"]),
        "masked_ce_mean": float(best["masked_ce_mean"]),
        "masked_acc_mean": float(best["masked_acc_mean"]),
    }


def run_gate_a_suite(
    *,
    algo: ContinuousEmbeddingDiffusion,
    tokenizer,
    cfg: OmegaConf,
    overfit_batches: list[Batch],
    steps: int,
    lr: float,
    grad_clip: float,
    autocast_bf16: bool,
    device: torch.device,
    out_dir: Path,
    recon_t_values: list[float],
    recon_low_t: float,
    recon_mid_t: float,
    recon_low_acc: float,
    recon_low_ce: float,
    recon_mid_acc: float,
    recon_high_acc_improve: float,
    recon_high_ce_improve: float,
    overfit_pre_acc_min: float,
    overfit_post_acc_min: float,
    overfit_fit_acc_min: float,
    roundtrip_fit_steps: int,
    roundtrip_fit_lr: float,
    train_embedding_in_gate_a: bool,
) -> dict[str, Any]:
    _stage("Gate A suite: A-Overfit + A-Recon")
    algo.train()
    params = [p for p in algo.denoiser.parameters() if p.requires_grad]
    if getattr(algo, "train_decoder_in_stage1", False) and getattr(algo, "embedding_provider_name", "") == "legacy_contextual":
        params.extend([p for p in algo.decoder.parameters() if p.requires_grad])
    if train_embedding_in_gate_a and getattr(algo, "embedding_provider_name", "") != "legacy_contextual":
        if isinstance(getattr(algo, "embedding_provider", None), torch.nn.Module):
            params.extend([p for p in algo.embedding_provider.parameters() if p.requires_grad])
    opt = torch.optim.AdamW(params, lr=lr)
    algo._trainer = SimpleNamespace(global_step=0, max_steps=steps, accumulate_grad_batches=1)

    curve = {"step": [], "loss": [], "ce": [], "masked_acc": []}
    jvp_failures = 0
    nan_failures = 0
    runtime_errors: list[str] = []

    pre = _evaluate_batches(algo, overfit_batches)
    pre_roundtrip = _evaluate_roundtrip_metrics(algo, overfit_batches, device=device)
    pre_recon = _evaluate_reconstruction_metrics(
        algo,
        overfit_batches,
        device=device,
        t_values=recon_t_values,
        seed=9001,
    )

    for step in range(int(steps)):
        batch = overfit_batches[step % len(overfit_batches)]
        b = Batch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            span_mask=batch.span_mask.to(device),
        )
        opt.zero_grad(set_to_none=True)
        try:
            with _autocast_context(device, enabled=autocast_bf16):
                loss, stats, _ = _compute_fixed_stage1_loss(algo, b)
        except RuntimeError as exc:
            msg = str(exc)
            runtime_errors.append(msg)
            if "torch.func.jvp failed" in msg:
                jvp_failures += 1
            break
        if not torch.isfinite(loss):
            nan_failures += 1
            runtime_errors.append(f"non-finite loss at step={step}: {float(loss.detach().cpu().item())}")
            break
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        opt.step()
        algo._trainer.global_step = step + 1

        curve["step"].append(step + 1)
        curve["loss"].append(float(loss.detach().cpu().item()))
        curve["ce"].append(float(stats["ce_loss"]))
        curve["masked_acc"].append(float(stats["masked_acc"]))

    post = _evaluate_batches(algo, overfit_batches)
    post_roundtrip = _evaluate_roundtrip_metrics(algo, overfit_batches, device=device)

    roundtrip_fit = {
        "performed": False,
        "reason": "disabled",
        "post": post_roundtrip,
        "curve": {"step": [], "ce": [], "acc": []},
    }
    recon_eval_model = algo
    if int(roundtrip_fit_steps) > 0:
        clone = ContinuousEmbeddingDiffusion(cfg, tokenizer).to(device)
        clone.load_state_dict(algo.state_dict())
        _ensure_jvp_safe_backend(clone)
        roundtrip_fit = _run_roundtrip_fit(
            clone,
            overfit_batches=overfit_batches,
            steps=int(roundtrip_fit_steps),
            lr=float(roundtrip_fit_lr),
            grad_clip=grad_clip,
            autocast_bf16=autocast_bf16,
            device=device,
        )
        recon_eval_model = clone

    post_recon = _evaluate_reconstruction_metrics(
        recon_eval_model,
        overfit_batches,
        device=device,
        t_values=recon_t_values,
        seed=9017,
    )

    pre_low = _lookup_t_metric(pre_recon, recon_low_t)
    pre_mid = _lookup_t_metric(pre_recon, recon_mid_t)
    post_low = _lookup_t_metric(post_recon, recon_low_t)
    post_mid = _lookup_t_metric(post_recon, recon_mid_t)

    pre_high = [x for x in pre_recon.get("per_t", []) if float(x["t"]) >= 0.5]
    post_high = [x for x in post_recon.get("per_t", []) if float(x["t"]) >= 0.5]
    pre_high_acc = float(np.mean([float(x["masked_acc_mean"]) for x in pre_high])) if pre_high else float("nan")
    post_high_acc = float(np.mean([float(x["masked_acc_mean"]) for x in post_high])) if post_high else float("nan")
    pre_high_ce = float(np.mean([float(x["masked_ce_mean"]) for x in pre_high])) if pre_high else float("nan")
    post_high_ce = float(np.mean([float(x["masked_ce_mean"]) for x in post_high])) if post_high else float("nan")

    recon_pass = bool(
        (post_low["masked_acc_mean"] >= recon_low_acc)
        and (post_low["masked_ce_mean"] <= recon_low_ce)
        and (post_mid["masked_acc_mean"] >= recon_mid_acc)
        and ((post_high_acc - pre_high_acc) >= recon_high_acc_improve)
        and ((pre_high_ce - post_high_ce) >= recon_high_ce_improve)
        and (len(post_recon.get("runtime_errors", [])) == 0)
    )

    fit_acc = float(roundtrip_fit.get("post", {}).get("roundtrip_acc_mean", float("nan")))
    overfit_pass = bool(
        (pre_roundtrip["roundtrip_acc_mean"] >= overfit_pre_acc_min)
        or (post_roundtrip["roundtrip_acc_mean"] >= overfit_post_acc_min)
        or (bool(roundtrip_fit.get("performed")) and fit_acc >= overfit_fit_acc_min)
    )

    train_sanity_pass = bool(
        (post["ce_mean"] < pre["ce_mean"])
        and (jvp_failures == 0)
        and (nan_failures == 0)
    )

    gate_overfit = {
        "gate": "A_overfit",
        "pass": overfit_pass,
        "pre_roundtrip": pre_roundtrip,
        "post_roundtrip": post_roundtrip,
        "roundtrip_fit": roundtrip_fit,
        "thresholds": {
            "pre_acc_min": float(overfit_pre_acc_min),
            "post_acc_min": float(overfit_post_acc_min),
            "fit_acc_min": float(overfit_fit_acc_min),
        },
    }
    gate_recon = {
        "gate": "A_recon",
        "pass": recon_pass,
        "pre": pre_recon,
        "post": post_recon,
        "pre_low": pre_low,
        "post_low": post_low,
        "pre_mid": pre_mid,
        "post_mid": post_mid,
        "pre_high_acc_mean": pre_high_acc,
        "post_high_acc_mean": post_high_acc,
        "pre_high_ce_mean": pre_high_ce,
        "post_high_ce_mean": post_high_ce,
        "thresholds": {
            "low_t": recon_low_t,
            "mid_t": recon_mid_t,
            "low_acc_min": recon_low_acc,
            "low_ce_max": recon_low_ce,
            "mid_acc_min": recon_mid_acc,
            "high_acc_improve_min": recon_high_acc_improve,
            "high_ce_improve_min": recon_high_ce_improve,
        },
    }
    gate_train = {
        "gate": "A_train",
        "pass": train_sanity_pass,
        "pre": pre,
        "post": post,
        "ce_drop": float(pre["ce_mean"] - post["ce_mean"]),
        "ce_drop_ratio": float(post["ce_mean"] / max(pre["ce_mean"], 1e-8)),
        "jvp_failures": int(jvp_failures),
        "nan_failures": int(nan_failures),
        "runtime_errors": runtime_errors,
        "curve": curve,
    }

    gate_suite = {
        "gate": "A_suite",
        "pass": bool(overfit_pass and recon_pass and train_sanity_pass),
        "overfit": gate_overfit,
        "recon": gate_recon,
        "train": gate_train,
    }

    _safe_json_dump(gate_train, out_dir / "gate_a_train.json")
    _safe_json_dump(gate_overfit, out_dir / "gate_a_overfit.json")
    _safe_json_dump(gate_recon, out_dir / "gate_a_recon.json")
    _safe_json_dump(gate_suite, out_dir / "gate_a_suite.json")
    _stage(
        f"Gate A done: overfit={gate_overfit['pass']} "
        f"recon={gate_recon['pass']} train={gate_train['pass']}"
    )
    return gate_suite


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _grad_stats(parameters) -> tuple[float, float]:
    sq_sum = 0.0
    max_abs = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq_sum += float(torch.sum(g.float() * g.float()).item())
        max_abs = max(max_abs, float(g.abs().max().item()))
    return float(math.sqrt(max(sq_sum, 0.0))), float(max_abs)


def _gate_b_worker(
    rank: int,
    world_size: int,
    payload: dict[str, Any],
) -> None:
    _set_seed(int(payload["seed"]))
    backend = payload["backend"]
    dist.init_process_group(
        backend=backend,
        init_method=payload["init_method"],
        rank=rank,
        world_size=world_size,
    )
    try:
        use_cuda = bool(payload["use_cuda"])
        if use_cuda:
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        tokenizer = AutoTokenizer.from_pretrained(payload["tokenizer_name"])
        _ensure_special_tokens(tokenizer)
        cfg = OmegaConf.create(payload["cfg"])
        algo = ContinuousEmbeddingDiffusion(cfg, tokenizer).to(device)
        algo.train()
        algo._trainer = SimpleNamespace(global_step=0, max_steps=1, accumulate_grad_batches=1)

        requested_backend = str(payload["attn_backend"])
        block_backends = sorted({str(b.attn_backend) for b in algo.denoiser.blocks})
        flash_supported = bool(supports_flash_attention(device))
        expected_kernel = "flash_attn" if (requested_backend in {"flash_attn"} and flash_supported) else "sdpa"
        if requested_backend == "auto":
            expected_kernel = "flash_attn" if flash_supported else "sdpa"
        # With attention_mask enabled, block path always uses SDPA masked kernels.
        masked_attention_forces_sdpa = True

        batch_blob = torch.load(payload["batch_path"], map_location="cpu")
        b = Batch(
            input_ids=batch_blob["input_ids"].to(device),
            attention_mask=batch_blob["attention_mask"].to(device),
            span_mask=batch_blob["span_mask"].to(device),
        )

        probe_info = _ensure_jvp_safe_backend(algo)
        jvp_ok = bool(probe_info.get("final_ok", True))
        jvp_msg = str(probe_info.get("message", ""))
        runtime_error = None

        params = [p for p in algo._get_parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(payload["lr"]))
        opt.zero_grad(set_to_none=True)
        try:
            with _autocast_context(device, enabled=bool(payload["autocast_bf16"])):
                loss, stats, _ = _compute_fixed_stage1_loss(algo, b)
        except RuntimeError as exc:
            runtime_error = str(exc)
            jvp_ok = False
            loss = torch.tensor(float("nan"), device=device)
            stats = {"masked_acc": float("nan"), "ce_loss": float("nan"), "main_loss": float("nan")}

        if runtime_error is None and torch.isfinite(loss):
            loss.backward()
            grad_norm, grad_abs_max = _grad_stats(params)
            # Manual all-reduce to mimic one DDP-synced optimization step for this custom objective path.
            for p in params:
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= float(world_size)
            opt.step()
        else:
            grad_norm, grad_abs_max = float("nan"), float("nan")

        local_stats = torch.tensor(
            [
                float(loss.detach().cpu().item()) if torch.isfinite(loss) else float("nan"),
                float(grad_norm),
                float(grad_abs_max),
                float(stats["masked_acc"]),
                float(stats["ce_loss"]),
            ],
            device=device,
            dtype=torch.float64,
        )
        gathered = [torch.zeros_like(local_stats) for _ in range(world_size)]
        dist.all_gather(gathered, local_stats)

        rank_json = {
            "rank": rank,
            "device": str(device),
            "jvp_ok": bool(jvp_ok),
            "jvp_message": str(jvp_msg),
            "jvp_probe": probe_info,
            "runtime_error": runtime_error,
            "local_stats": {
                "loss": float(local_stats[0].item()),
                "grad_norm": float(local_stats[1].item()),
                "grad_abs_max": float(local_stats[2].item()),
                "masked_acc": float(local_stats[3].item()),
                "ce_loss": float(local_stats[4].item()),
            },
            "all_ranks_stats": [[float(x.item()) for x in t] for t in gathered],
            "attention_backend": {
                "requested": requested_backend,
                "block_backends": block_backends,
                "flash_supported": flash_supported,
                "expected_kernel": expected_kernel,
                "masked_attention_forces_sdpa": masked_attention_forces_sdpa,
            },
        }
        out_file = Path(payload["out_dir"]) / f"gate_b_rank{rank}.json"
        _safe_json_dump(rank_json, out_file)
    finally:
        dist.destroy_process_group()


def run_gate_b(
    *,
    cfg: OmegaConf,
    tokenizer_name: str,
    batch: Batch,
    out_dir: Path,
    device: torch.device,
    seed: int,
    world_size: int,
    lr: float,
    autocast_bf16: bool,
    attn_backend: str,
) -> dict[str, Any]:
    _stage("Gate B: JVP/DDP backend coverage")
    requested_world_size = int(world_size)
    use_cuda = bool(device.type == "cuda" and torch.cuda.is_available())
    if use_cuda and world_size > torch.cuda.device_count():
        world_size = torch.cuda.device_count()
    if world_size < 2:
        world_size = 2 if not use_cuda else min(2, torch.cuda.device_count())
    backend = "nccl" if use_cuda else "gloo"

    batch_path = out_dir / "gate_b_batch.pt"
    torch.save(
        {
            "input_ids": batch.input_ids.cpu(),
            "attention_mask": batch.attention_mask.cpu(),
            "span_mask": batch.span_mask.cpu(),
        },
        batch_path,
    )

    port = _find_free_port()
    payload = {
        "seed": int(seed),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "tokenizer_name": tokenizer_name,
        "batch_path": str(batch_path),
        "out_dir": str(out_dir),
        "init_method": f"tcp://127.0.0.1:{port}",
        "backend": backend,
        "use_cuda": bool(use_cuda),
        "lr": float(lr),
        "autocast_bf16": bool(autocast_bf16),
        "attn_backend": str(attn_backend),
    }

    spawn_error = None
    try:
        mp.spawn(
            _gate_b_worker,
            args=(int(world_size), payload),
            nprocs=int(world_size),
            join=True,
        )
    except Exception as exc:  # pragma: no cover - exercised in integration runs
        spawn_error = str(exc)

    rank_files = sorted(out_dir.glob("gate_b_rank*.json"))
    rank_reports = [_safe_load_json(p) for p in rank_files]
    any_runtime_error = bool(spawn_error) or any(r.get("runtime_error") for r in rank_reports)
    any_jvp_fail = bool(spawn_error) or any(not bool(r.get("jvp_ok", False)) for r in rank_reports)

    # Rank-consistency checks.
    all_stats = [r["local_stats"] for r in rank_reports]
    keys = ["loss", "grad_norm", "grad_abs_max", "masked_acc", "ce_loss"]
    max_diffs = {}
    if all_stats:
        for k in keys:
            vals = np.array([float(s[k]) for s in all_stats], dtype=np.float64)
            max_diffs[k] = float(np.nanmax(vals) - np.nanmin(vals))
    else:
        for k in keys:
            max_diffs[k] = float("nan")

    tol = 5e-4
    rank_consistent = bool(all((not math.isnan(v)) and (v <= tol) for v in max_diffs.values()))

    ddp_validated = bool(world_size >= 2)
    pass_gate = bool((not any_runtime_error) and (not any_jvp_fail) and rank_consistent and ddp_validated)
    out = {
        "gate": "B",
        "pass": pass_gate,
        "backend": backend,
        "requested_world_size": requested_world_size,
        "world_size": int(world_size),
        "ddp_validated": ddp_validated,
        "rank_consistency_tolerance": tol,
        "max_diffs": max_diffs,
        "any_runtime_error": any_runtime_error,
        "any_jvp_fail": any_jvp_fail,
        "spawn_error": spawn_error,
        "rank_files": [str(p) for p in rank_files],
    }
    _safe_json_dump(out, out_dir / "gate_b_jvp_ddp.json")
    _stage(f"Gate B done: pass={out['pass']}")
    return out


def _safe_load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_gate_b_cuda_requirement(
    *,
    gate_b: dict[str, Any],
    device: torch.device,
    require_b_cuda: bool,
    artifact_path: str,
) -> dict[str, Any]:
    if not require_b_cuda:
        return {
            "gate": "B_cuda",
            "pass": True,
            "required": False,
            "reason": "require_b_cuda_disabled",
        }

    local_ok = bool(
        device.type == "cuda"
        and gate_b.get("pass", False)
        and str(gate_b.get("backend", "")) == "nccl"
        and int(gate_b.get("world_size", 0)) >= 2
    )
    if local_ok:
        return {
            "gate": "B_cuda",
            "pass": True,
            "required": True,
            "source": "local_run",
            "backend": gate_b.get("backend"),
            "world_size": gate_b.get("world_size"),
        }

    artifact = None
    if artifact_path:
        p = Path(artifact_path)
        if p.exists():
            artifact = _safe_load_json(p)
    artifact_ok = bool(
        artifact is not None
        and artifact.get("pass", False)
        and str(artifact.get("backend", "")) == "nccl"
        and int(artifact.get("world_size", 0)) >= 2
    )
    if artifact_ok:
        return {
            "gate": "B_cuda",
            "pass": True,
            "required": True,
            "source": "artifact",
            "artifact_path": artifact_path,
            "backend": artifact.get("backend"),
            "world_size": artifact.get("world_size"),
        }

    return {
        "gate": "B_cuda",
        "pass": False,
        "required": True,
        "source": "none",
        "local_gate_b": gate_b,
        "artifact_path": artifact_path if artifact_path else None,
        "artifact_found": artifact is not None,
    }


def run_gate_c(
    *,
    algo: ContinuousEmbeddingDiffusion,
    tokenizer,
    batch: Batch,
    device: torch.device,
    out_dir: Path,
    eps_tolerance: float,
) -> dict[str, Any]:
    _stage("Gate C: clamp invariance")
    algo.eval()
    b = Batch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        span_mask=batch.span_mask.to(device),
    )
    conditioning_ids = _build_conditioning_ids(tokenizer, b)
    fixed = (~b.span_mask.bool()) & b.attention_mask.bool()
    z_known = algo._embed_inputs(conditioning_ids, b.attention_mask)

    checks: list[dict[str, Any]] = []
    runtime_errors: list[str] = []
    for steps in [1, 2, 4]:
        try:
            gen = torch.Generator(device=device).manual_seed(3000 + steps)
            z_hat, pred_ids = _sample_conditioned_embeddings(
                algo,
                conditioning_ids=conditioning_ids,
                span_mask=b.span_mask,
                attention_mask=b.attention_mask,
                num_steps=steps,
                sampling_method="ddpm",
                eta=0.0,
                eps=1e-5,
                generator=gen,
            )
            ids_equal = bool(torch.equal(pred_ids[fixed], b.input_ids[fixed])) if int(fixed.sum().item()) > 0 else True
            drift = ((z_hat - z_known).pow(2).sum(dim=-1).sqrt())
            drift_max = float(drift[fixed].max().item()) if int(fixed.sum().item()) > 0 else 0.0
            checks.append(
                {
                    "num_steps": steps,
                    "unmasked_ids_bitwise_equal": ids_equal,
                    "unmasked_embedding_l2_drift_max": drift_max,
                    "pass": bool(ids_equal and drift_max <= eps_tolerance),
                }
            )
        except RuntimeError as exc:
            runtime_errors.append(f"steps={steps}: {exc}")
            checks.append(
                {
                    "num_steps": steps,
                    "unmasked_ids_bitwise_equal": False,
                    "unmasked_embedding_l2_drift_max": float("nan"),
                    "pass": False,
                }
            )

    gate_pass = all(c["pass"] for c in checks)
    out = {
        "gate": "C",
        "pass": bool(gate_pass),
        "eps_tolerance": float(eps_tolerance),
        "checks": checks,
        "runtime_errors": runtime_errors,
    }
    _safe_json_dump(out, out_dir / "gate_c_clamp_invariance.json")
    _stage(f"Gate C done: pass={out['pass']}")
    return out


def _module_grad_norm(module: Optional[torch.nn.Module]) -> tuple[float, int]:
    if module is None:
        return 0.0, 0
    sq_sum = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sq_sum += float(torch.sum(g * g).item())
        count += int(torch.count_nonzero(g).item())
    return float(math.sqrt(max(sq_sum, 0.0))), int(count)


def run_gate_f(
    *,
    algo: ContinuousEmbeddingDiffusion,
    batch: Batch,
    device: torch.device,
    out_dir: Path,
    denoiser_grad_norm_min: float,
) -> dict[str, Any]:
    _stage("Gate F: CE gradient-path sanity")
    algo.train()
    b = Batch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        span_mask=batch.span_mask.to(device),
    )
    for p in algo.parameters():
        if p.grad is not None:
            p.grad = None

    runtime_error = None
    x0_hat_source = "unknown"
    try:
        z0 = algo._embed_inputs(b.input_ids, b.attention_mask)
        out = algo.objective.compute(
            model=algo,
            x0=z0,
            attention_mask=b.attention_mask,
            span_mask=b.span_mask,
            input_ids=b.input_ids,
        )
        x0_hat = out.x0_hat if out.x0_hat is not None else z0
        x0_hat_source = "objective" if out.x0_hat is not None else "fallback_z0"
        ce_only = algo._compute_ce_loss(
            x0_hat,
            input_ids=b.input_ids,
            attention_mask=b.attention_mask,
            token_mask=b.span_mask,
        )
        ce_only.backward()
    except RuntimeError as exc:
        runtime_error = str(exc)
        ce_only = torch.tensor(float("nan"), device=device)

    denoiser_grad_norm, denoiser_nz = _module_grad_norm(algo.denoiser)
    decoder_grad_norm, decoder_nz = _module_grad_norm(algo.decoder)
    if isinstance(getattr(algo, "embedding_provider", None), torch.nn.Module):
        embedding_grad_norm, embedding_nz = _module_grad_norm(algo.embedding_provider)
    else:
        embedding_grad_norm, embedding_nz = 0.0, 0

    pass_gate = bool(
        (runtime_error is None)
        and torch.isfinite(ce_only)
        and (x0_hat_source == "objective")
        and (denoiser_grad_norm >= float(denoiser_grad_norm_min))
    )

    out = {
        "gate": "F",
        "pass": pass_gate,
        "runtime_error": runtime_error,
        "x0_hat_source": x0_hat_source,
        "ce_loss": float(ce_only.detach().cpu().item()) if torch.isfinite(ce_only) else float("nan"),
        "ce_grad_norm": {
            "denoiser": denoiser_grad_norm,
            "decoder": decoder_grad_norm,
            "embedding": embedding_grad_norm,
        },
        "ce_grad_nnz": {
            "denoiser": denoiser_nz,
            "decoder": decoder_nz,
            "embedding": embedding_nz,
        },
        "thresholds": {
            "denoiser_grad_norm_min": float(denoiser_grad_norm_min),
        },
    }
    _safe_json_dump(out, out_dir / "gate_f_ce_grad_path.json")
    _stage(f"Gate F done: pass={out['pass']}")
    return out


def run_gate_g(
    *,
    algo: ContinuousEmbeddingDiffusion,
    batch: Batch,
    device: torch.device,
    out_dir: Path,
    zero_loss_tol: float,
    all_one_match_tol: float,
) -> dict[str, Any]:
    _stage("Gate G: mask semantics sanity (G1/G2)")
    algo.eval()
    b = Batch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        span_mask=batch.span_mask.to(device),
    )
    runtime_errors: list[str] = []

    try:
        zero_mask = torch.zeros_like(b.attention_mask)
        total0, stats0, _ = _compute_stage1_losses(
            algo,
            input_ids=b.input_ids,
            attention_mask=b.attention_mask,
            span_mask=zero_mask,
            ce_token_mask=zero_mask,
        )
        g1 = {
            "main_loss": float(stats0["main_loss"]),
            "ce_loss": float(stats0["ce_loss"]),
            "total_loss": float(total0.detach().cpu().item()),
        }
        g1_pass = bool(
            abs(g1["main_loss"]) <= zero_loss_tol
            and abs(g1["ce_loss"]) <= zero_loss_tol
            and abs(g1["total_loss"]) <= zero_loss_tol
        )
    except RuntimeError as exc:
        runtime_errors.append(f"G1: {exc}")
        g1 = {"main_loss": float("nan"), "ce_loss": float("nan"), "total_loss": float("nan")}
        g1_pass = False

    try:
        one_mask = b.attention_mask.bool().to(dtype=b.attention_mask.dtype)
        cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(8123)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(8123)
            total1, stats1, _ = _compute_stage1_losses(
                algo,
                input_ids=b.input_ids,
                attention_mask=b.attention_mask,
                span_mask=one_mask,
                ce_token_mask=one_mask,
            )
            torch.manual_seed(8123)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(8123)
            total_u, stats_u, _ = _compute_stage1_losses(
                algo,
                input_ids=b.input_ids,
                attention_mask=b.attention_mask,
                span_mask=None,
                ce_token_mask=None,
            )
        diff_main = abs(float(stats1["main_loss"]) - float(stats_u["main_loss"]))
        diff_ce = abs(float(stats1["ce_loss"]) - float(stats_u["ce_loss"]))
        diff_total = abs(float(total1.detach().cpu().item()) - float(total_u.detach().cpu().item()))
        ref_total = max(abs(float(total_u.detach().cpu().item())), 1.0)
        g2 = {
            "all_one": {
                "main_loss": float(stats1["main_loss"]),
                "ce_loss": float(stats1["ce_loss"]),
                "total_loss": float(total1.detach().cpu().item()),
            },
            "unconditional": {
                "main_loss": float(stats_u["main_loss"]),
                "ce_loss": float(stats_u["ce_loss"]),
                "total_loss": float(total_u.detach().cpu().item()),
            },
            "diff": {
                "main_loss_abs": diff_main,
                "ce_loss_abs": diff_ce,
                "total_loss_abs": diff_total,
                "total_loss_rel": diff_total / ref_total,
            },
        }
        g2_pass = bool((diff_total / ref_total) <= all_one_match_tol)
    except RuntimeError as exc:
        runtime_errors.append(f"G2: {exc}")
        g2 = {}
        g2_pass = False

    out = {
        "gate": "G",
        "pass": bool(g1_pass and g2_pass),
        "G1_all_zero_mask": {"pass": g1_pass, **g1},
        "G2_all_one_mask": {"pass": g2_pass, **g2},
        "thresholds": {
            "zero_loss_tol": float(zero_loss_tol),
            "all_one_match_tol": float(all_one_match_tol),
        },
        "runtime_errors": runtime_errors,
    }
    _safe_json_dump(out, out_dir / "gate_g_mask_semantics.json")
    _stage(f"Gate G done: pass={out['pass']}")
    return out


def run_gate_d(
    *,
    algo: ContinuousEmbeddingDiffusion,
    cfg: OmegaConf,
    tokenizer,
    dev_batches: list[Batch],
    train_batches: list[Batch],
    device: torch.device,
    out_dir: Path,
    baseline_metrics: Optional[dict[str, float]],
    signal_ce_drop_ratio_min: float,
    signal_acc_gain_min: float,
    pilot_steps: int,
    pilot_lr: float,
    pilot_loss_mode: str,
    grad_clip: float,
    autocast_bf16: bool,
    pilot_init_state: Optional[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    _stage("Gate D: deterministic internal metrics (stability + signal)")
    def _eval_dev_metrics(model: ContinuousEmbeddingDiffusion, seed_offset: int) -> tuple[dict[str, float], list[str]]:
        model.eval()
        ce_vals: list[float] = []
        acc_vals: list[float] = []
        mismatch_vals: list[float] = []
        drift_vals: list[float] = []
        roundtrip_mismatch_vals: list[float] = []
        errors: list[str] = []

        with torch.no_grad():
            for i, b0 in enumerate(dev_batches):
                try:
                    b = Batch(
                        input_ids=b0.input_ids.to(device),
                        attention_mask=b0.attention_mask.to(device),
                        span_mask=b0.span_mask.to(device),
                    )
                    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(int(seed_offset + i))
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(int(seed_offset + i))
                        _, stats, x0_hat = _compute_fixed_stage1_loss(model, b)
                    logits = model._decode_logits(x0_hat)
                    pred_ids = logits.argmax(dim=-1)
                    ce = float(stats["ce_loss"])
                    acc = float(stats["masked_acc"])

                    valid = b.attention_mask.bool() & b.span_mask.bool()
                    mismatch = (pred_ids != b.input_ids) & valid
                    mismatch_rate = float(mismatch.float().mean().item())

                    re_emb = model._embed_inputs(pred_ids, b.attention_mask)
                    emb_drift = ((re_emb - x0_hat).pow(2).sum(dim=-1).sqrt())
                    valid_all = b.attention_mask.bool()
                    emb_drift_mean = float(emb_drift[valid_all].mean().item()) if int(valid_all.sum().item()) > 0 else 0.0

                    red_logits = model._decode_logits(re_emb)
                    red_ids = red_logits.argmax(dim=-1)
                    rt_mismatch = (red_ids != pred_ids) & valid_all
                    rt_mismatch_rate = float(rt_mismatch.float().mean().item())

                    ce_vals.append(float(ce))
                    acc_vals.append(float(acc))
                    mismatch_vals.append(mismatch_rate)
                    drift_vals.append(emb_drift_mean)
                    roundtrip_mismatch_vals.append(rt_mismatch_rate)
                except RuntimeError as exc:
                    errors.append(f"batch={i}: {exc}")

        metrics = {
            "masked_ce_mean": float(np.mean(ce_vals)) if ce_vals else float("nan"),
            "masked_token_acc_mean": float(np.mean(acc_vals)) if acc_vals else float("nan"),
            "token_mismatch_rate_mean": float(np.mean(mismatch_vals)) if mismatch_vals else float("nan"),
            "embedding_roundtrip_drift_l2_mean": float(np.mean(drift_vals)) if drift_vals else float("nan"),
            "roundtrip_token_mismatch_rate_mean": float(np.mean(roundtrip_mismatch_vals)) if roundtrip_mismatch_vals else float("nan"),
        }
        return metrics, errors

    fixed_dev_metrics, runtime_errors = _eval_dev_metrics(algo, seed_offset=4100)

    signal = {
        "has_baseline": baseline_metrics is not None,
        "baseline": baseline_metrics,
        "ce_drop_ratio": float("nan"),
        "acc_gain": float("nan"),
        "pass": False,
        "pilot_steps": int(pilot_steps),
        "pilot_lr": float(pilot_lr),
        "pilot_loss_mode": str(pilot_loss_mode),
        "pilot_runtime_errors": [],
    }
    signal_metrics = fixed_dev_metrics
    signal_errors: list[str] = []
    if baseline_metrics is not None:
        if int(pilot_steps) > 0 and len(train_batches) > 0:
            pilot = ContinuousEmbeddingDiffusion(cfg, tokenizer).to(device)
            if pilot_init_state is not None:
                pilot.load_state_dict(pilot_init_state)
            else:
                pilot.load_state_dict(algo.state_dict())
            _ensure_jvp_safe_backend(pilot)
            pilot.train()
            params = [p for p in pilot._get_parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=float(pilot_lr))
            pilot._trainer = SimpleNamespace(global_step=0, max_steps=int(pilot_steps), accumulate_grad_batches=1)
            pilot_curve = {"step": [], "loss": [], "ce_loss": [], "masked_acc": []}
            for step in range(int(pilot_steps)):
                b0 = train_batches[step % len(train_batches)]
                b = Batch(
                    input_ids=b0.input_ids.to(device),
                    attention_mask=b0.attention_mask.to(device),
                    span_mask=b0.span_mask.to(device),
                )
                opt.zero_grad(set_to_none=True)
                try:
                    with _autocast_context(device, enabled=autocast_bf16):
                        if str(pilot_loss_mode).lower() == "ce_only":
                            z0 = pilot._embed_inputs(b.input_ids, b.attention_mask)
                            out = pilot.objective.compute(
                                model=pilot,
                                x0=z0,
                                attention_mask=b.attention_mask,
                                span_mask=b.span_mask,
                                input_ids=b.input_ids,
                            )
                            x0_hat = out.x0_hat if out.x0_hat is not None else z0
                            ce = pilot._compute_ce_loss(
                                x0_hat,
                                input_ids=b.input_ids,
                                attention_mask=b.attention_mask,
                                token_mask=b.span_mask,
                            )
                            loss = ce
                            logits = pilot._decode_logits(x0_hat)
                            stats = {
                                "ce_loss": float(ce.detach().cpu().item()),
                                "masked_acc": _masked_accuracy(logits.detach(), b.input_ids, b.span_mask, b.attention_mask),
                            }
                        else:
                            loss, stats, _ = _compute_fixed_stage1_loss(pilot, b)
                except RuntimeError as exc:
                    signal["pilot_runtime_errors"].append(f"step={step}: {exc}")
                    break
                if not torch.isfinite(loss):
                    signal["pilot_runtime_errors"].append(f"step={step}: non-finite loss")
                    break
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))
                opt.step()
                pilot._trainer.global_step = step + 1
                pilot_curve["step"].append(step + 1)
                pilot_curve["loss"].append(float(loss.detach().cpu().item()))
                pilot_curve["ce_loss"].append(float(stats["ce_loss"]))
                pilot_curve["masked_acc"].append(float(stats["masked_acc"]))
            signal["pilot_curve"] = pilot_curve
            signal_metrics, signal_eval_errors = _eval_dev_metrics(pilot, seed_offset=5100)
            signal_errors = list(signal_eval_errors)
            signal["pilot_runtime_errors"].extend(signal_eval_errors)
        signal["post_metrics"] = signal_metrics
        base_ce = float(baseline_metrics.get("masked_ce_mean", float("nan")))
        base_acc = float(baseline_metrics.get("masked_token_acc_mean", float("nan")))
        if np.isfinite(base_ce) and np.isfinite(base_acc):
            ce_drop_ratio = (base_ce - signal_metrics["masked_ce_mean"]) / max(abs(base_ce), 1e-8)
            acc_gain = signal_metrics["masked_token_acc_mean"] - base_acc
            signal["ce_drop_ratio"] = float(ce_drop_ratio)
            signal["acc_gain"] = float(acc_gain)
            signal["pass"] = bool(
                (ce_drop_ratio >= float(signal_ce_drop_ratio_min))
                and (acc_gain >= float(signal_acc_gain_min))
            )

    stability_metrics = signal_metrics if int(pilot_steps) > 0 and len(train_batches) > 0 else fixed_dev_metrics
    stability_errors = signal_errors if int(pilot_steps) > 0 and len(train_batches) > 0 else runtime_errors
    stability_pass = bool(
        len(stability_errors) == 0
        and np.isfinite(stability_metrics["masked_ce_mean"])
        and np.isfinite(stability_metrics["masked_token_acc_mean"])
        and np.isfinite(stability_metrics["embedding_roundtrip_drift_l2_mean"])
        and (stability_metrics["masked_ce_mean"] < 25.0)
        and (stability_metrics["embedding_roundtrip_drift_l2_mean"] < 6.0)
    )

    out = {
        "gate": "D",
        "pass": bool(stability_pass and signal["pass"]),
        "D_stability": {
            "pass": stability_pass,
            "fixed_dev_metrics": stability_metrics,
            "runtime_errors": stability_errors,
        },
        "D_signal": {
            **signal,
            "thresholds": {
                "ce_drop_ratio_min": float(signal_ce_drop_ratio_min),
                "acc_gain_min": float(signal_acc_gain_min),
            },
        },
        "fixed_dev_metrics_post_gate_a": fixed_dev_metrics,
        "runtime_errors_post_gate_a": runtime_errors,
    }
    _safe_json_dump(out, out_dir / "gate_d_internal_metrics.json")
    _stage(
        f"Gate D done: pass={out['pass']} "
        f"(stability={out['D_stability']['pass']}, signal={out['D_signal']['pass']})"
    )
    return out


def _current_rss_mb() -> float:
    # Linux ru_maxrss is KB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _profile_variant(
    *,
    cfg: OmegaConf,
    tokenizer,
    batch: Batch,
    device: torch.device,
    seed: int,
    steps: int,
    lr: float,
    autocast_bf16: bool,
) -> dict[str, Any]:
    _set_seed(seed)
    algo = ContinuousEmbeddingDiffusion(cfg, tokenizer).to(device)
    jvp_probe = _ensure_jvp_safe_backend(algo)
    algo.train()
    algo._trainer = SimpleNamespace(global_step=0, max_steps=steps, accumulate_grad_batches=1)

    params = [p for p in algo._get_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    step_times = []
    peak_mem_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    warmup = 1
    total_steps = warmup + int(steps)
    runtime_error = None
    for i in range(total_steps):
        b = Batch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            span_mask=batch.span_mask.to(device),
        )
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        try:
            with _autocast_context(device, enabled=autocast_bf16):
                loss, _, _ = _compute_fixed_stage1_loss(algo, b)
        except RuntimeError as exc:
            runtime_error = str(exc)
            break
        if not torch.isfinite(loss):
            runtime_error = "non-finite loss in profile variant"
            break
        loss.backward()
        opt.step()
        t1 = time.perf_counter()
        algo._trainer.global_step += 1
        if i >= warmup:
            step_times.append(float(t1 - t0))
        if device.type == "cuda":
            peak_mem_mb = max(
                peak_mem_mb,
                float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)),
            )
        else:
            peak_mem_mb = max(peak_mem_mb, _current_rss_mb())

    mean_step_time = float(np.mean(step_times)) if step_times else float("nan")
    tokens_per_step = int(batch.input_ids.shape[0] * batch.input_ids.shape[1])
    tokens_per_sec = float(tokens_per_step / mean_step_time) if step_times and mean_step_time > 0 else float("nan")
    return {
        "jvp_probe": jvp_probe,
        "runtime_error": runtime_error,
        "mean_step_time_sec": mean_step_time,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_mb": float(peak_mem_mb),
        "timed_steps": int(len(step_times)),
        "tokens_per_step": int(tokens_per_step),
    }


def run_gate_e(
    *,
    cfg_base: OmegaConf,
    tokenizer,
    batch: Batch,
    device: torch.device,
    out_dir: Path,
    seed: int,
    steps: int,
    lr: float,
    autocast_bf16: bool,
    objective_name: str,
    e2_min_tps_ratio: float,
    e2_target_tokens: float,
) -> dict[str, Any]:
    _stage("Gate E: throughput/memory profiling")
    profiles = []
    for grad_ckpt in [False, True]:
        cfg = OmegaConf.create(OmegaConf.to_container(cfg_base, resolve=True))
        cfg.model.gradient_checkpointing = bool(grad_ckpt)
        prof = _profile_variant(
            cfg=cfg,
            tokenizer=tokenizer,
            batch=batch,
            device=device,
            seed=seed + (13 if grad_ckpt else 7),
            steps=steps,
            lr=lr,
            autocast_bf16=autocast_bf16,
        )
        prof["gradient_checkpointing"] = bool(grad_ckpt)
        profiles.append(prof)

    pass_gate = all((p["runtime_error"] is None) for p in profiles)
    e2 = {
        "pass": True,
        "not_applicable": True,
    }
    if objective_name in {"imf", "meanflow"}:
        flow_cfg = OmegaConf.create(OmegaConf.to_container(cfg_base, resolve=True))
        flow_cfg.algo.objective = "flow_matching"
        flow_cfg.algo.interpolant = "rectified"
        flow_cfg.model.objective = "flow_matching"
        flow_cfg.model.gradient_checkpointing = False
        flow_prof = _profile_variant(
            cfg=flow_cfg,
            tokenizer=tokenizer,
            batch=batch,
            device=device,
            seed=seed + 101,
            steps=steps,
            lr=lr,
            autocast_bf16=autocast_bf16,
        )
        current_prof = next((p for p in profiles if not bool(p["gradient_checkpointing"])), profiles[0])
        current_tps = float(current_prof["tokens_per_sec"])
        flow_tps = float(flow_prof["tokens_per_sec"])
        ratio = float(current_tps / max(flow_tps, 1e-8)) if np.isfinite(current_tps) and np.isfinite(flow_tps) else float("nan")
        current_hours = float(e2_target_tokens / max(current_tps, 1e-8) / 3600.0) if np.isfinite(current_tps) else float("nan")
        flow_hours = float(e2_target_tokens / max(flow_tps, 1e-8) / 3600.0) if np.isfinite(flow_tps) else float("nan")
        e2 = {
            "pass": bool(
                current_prof["runtime_error"] is None
                and flow_prof["runtime_error"] is None
                and np.isfinite(ratio)
                and (ratio >= float(e2_min_tps_ratio))
            ),
            "not_applicable": False,
            "thresholds": {
                "min_imf_to_flow_tps_ratio": float(e2_min_tps_ratio),
                "target_tokens_for_budget": float(e2_target_tokens),
            },
            "imf_or_meanflow_no_gc_profile": current_prof,
            "flow_matching_no_gc_profile": flow_prof,
            "imf_to_flow_tps_ratio": ratio,
            "projected_hours": {
                "imf_or_meanflow": current_hours,
                "flow_matching": flow_hours,
            },
        }

    out = {
        "gate": "E",
        "pass": bool(pass_gate),
        "profiles": profiles,
        "E2": e2,
    }
    _safe_json_dump(out, out_dir / "gate_e_throughput_memory.json")
    _stage(f"Gate E done: pass={out['pass']}")
    return out


def _plot_gate_a_curve(gate_a: dict[str, Any], out_path: Path) -> None:
    curve = gate_a.get("curve", {})
    if not curve and isinstance(gate_a.get("train"), dict):
        curve = gate_a["train"].get("curve", {})
    x = curve.get("step", [])
    if not x:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(x, curve.get("ce", []), label="masked CE", color="#1f77b4")
    axes[0].plot(x, curve.get("loss", []), label="total loss", color="#ff7f0e")
    axes[0].set_title("Gate A Overfit Loss")
    axes[0].set_xlabel("Step")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, curve.get("masked_acc", []), color="#2ca02c")
    axes[1].set_title("Gate A Masked Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_gate_status(summary: dict[str, Any], out_path: Path) -> None:
    preferred_order = [
        "A_overfit",
        "A_recon",
        "A_train",
        "B",
        "B_cuda",
        "C",
        "D_stability",
        "D_signal",
        "F",
        "G",
        "E",
        "E2",
    ]
    labels = [k for k in preferred_order if k in summary["gates"]]
    if not labels:
        labels = list(summary["gates"].keys())
    vals = [1.0 if bool(summary["gates"][k]["pass"]) else 0.0 for k in labels]
    colors = ["#2ca02c" if v > 0.5 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(max(7.2, 0.7 * len(labels)), 3.6))
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Pass (1) / Fail (0)")
    ax.set_title("Pre-Scale Gates A-E")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, "PASS" if v > 0.5 else "FAIL", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_summary_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Pre-Scale Gates Study (A-E)",
        "",
        f"- Timestamp (UTC): {summary['timestamp_utc']}",
        f"- Device: `{summary['runtime']['device']}`",
        f"- Objective: `{summary['runtime']['objective']}` / Interpolant: `{summary['runtime']['interpolant']}`",
        f"- OWT examples used: {summary['runtime']['owt_examples_used']}",
        "",
        "## Gate Status",
        "",
    ]
    for k, v in summary["gates"].items():
        lines.append(f"- Gate {k}: {'PASS' if summary['gates'][k]['pass'] else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Gate A is split into A-Overfit (roundtrip embedding/readout sanity) and A-Recon (forward-noise reconstruction).",
            "- Gate B checks JVP + multi-rank consistency with explicit backend diagnostics.",
            "- Gate B-CUDA requires a true multi-GPU CUDA validation (local or via artifact) before scaling.",
            "- Gate C enforces infill clamp invariance on unmasked positions.",
            "- Gate D reports deterministic internal quality metrics split into D-Stability and D-Signal.",
            "- Gate F verifies CE gradient flow reaches the denoiser.",
            "- Gate G verifies all-zero/all-one span-mask semantics.",
            "- Gate E measures training throughput/memory with JVP enabled and includes E2 objective-comparison profiling.",
            "",
            "## Artifacts",
            "",
            "- `gate_a_train.json`",
            "- `gate_a_overfit.json`",
            "- `gate_a_recon.json`",
            "- `gate_a_suite.json`",
            "- `gate_b_jvp_ddp.json` and `gate_b_rank*.json`",
            "- `gate_b_cuda_requirement.json`",
            "- `gate_c_clamp_invariance.json`",
            "- `gate_d_internal_metrics.json`",
            "- `gate_f_ce_grad_path.json`",
            "- `gate_g_mask_semantics.json`",
            "- `gate_e_throughput_memory.json`",
            "- `gate_status.png`",
            "- `gate_a_curve.png`",
            "- `summary.json`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--tokenizer-name", type=str, default="gpt2")
    p.add_argument("--owt-cache-dir", type=str, default="")
    p.add_argument("--owt-examples", type=int, default=320)
    p.add_argument("--synthetic-data", action="store_true")
    p.add_argument("--synthetic-vocab-size", type=int, default=8192)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--dev-batches", type=int, default=4)
    p.add_argument("--overfit-batches", type=int, default=1)
    p.add_argument("--span-patterns", type=int, default=3)

    p.add_argument("--objective", type=str, default="meanflow", choices=["ddpm", "flow_matching", "meanflow", "imf"])
    p.add_argument("--interpolant", type=str, default="rectified", choices=["vp", "rectified"])
    p.add_argument("--embedding-provider", type=str, default="tied", choices=["lookup", "tied", "legacy_contextual"])
    p.add_argument("--embedding-trainable", action="store_true")
    p.add_argument("--attn-backend", type=str, default="sdpa", choices=["auto", "sdpa", "flash_attn"])

    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--train-timesteps", type=int, default=128)

    p.add_argument("--lambda-ce", type=float, default=0.2)
    p.add_argument("--self-conditioning", action="store_true")
    p.add_argument("--self-conditioning-prob", type=float, default=0.5)
    p.add_argument("--imf-alpha", type=float, default=1.0)
    p.add_argument("--imf-warmup-steps", type=int, default=40)
    p.add_argument("--imf-ramp-steps", type=int, default=80)
    p.add_argument("--min-snr-gamma", type=float, default=5.0)

    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--autocast-bf16", action="store_true")
    p.add_argument("--gate-a-steps", type=int, default=700)
    p.add_argument("--gate-a-recon-t-values", type=str, default="0.05,0.2,0.5,0.8")
    p.add_argument("--gate-a-recon-low-t", type=float, default=0.05)
    p.add_argument("--gate-a-recon-mid-t", type=float, default=0.2)
    p.add_argument("--gate-a-recon-low-acc", type=float, default=0.98)
    p.add_argument("--gate-a-recon-low-ce", type=float, default=0.2)
    p.add_argument("--gate-a-recon-mid-acc", type=float, default=0.90)
    p.add_argument("--gate-a-recon-high-acc-improve", type=float, default=0.02)
    p.add_argument("--gate-a-recon-high-ce-improve", type=float, default=0.05)
    p.add_argument("--gate-a-overfit-pre-acc-min", type=float, default=0.90)
    p.add_argument("--gate-a-overfit-post-acc-min", type=float, default=0.95)
    p.add_argument("--gate-a-overfit-fit-acc-min", type=float, default=0.99)
    p.add_argument("--gate-a-roundtrip-fit-steps", type=int, default=250)
    p.add_argument("--gate-a-roundtrip-fit-lr", type=float, default=5e-3)
    p.add_argument("--gate-a-train-embedding", action="store_true")
    p.add_argument("--gate-b-world-size", type=int, default=2)
    p.add_argument("--require-b-cuda", action="store_true")
    p.add_argument("--gate-b-cuda-artifact", type=str, default="")
    p.add_argument("--gate-c-drift-tol", type=float, default=1e-6)
    p.add_argument("--gate-d-signal-ce-drop-ratio-min", type=float, default=0.10)
    p.add_argument("--gate-d-signal-acc-gain-min", type=float, default=0.02)
    p.add_argument("--gate-d-train-batches", type=int, default=32)
    p.add_argument("--gate-d-pilot-steps", type=int, default=250)
    p.add_argument("--gate-d-pilot-lr", type=float, default=3e-3)
    p.add_argument("--gate-d-pilot-loss-mode", type=str, default="ce_only", choices=["ce_only", "total"])
    p.add_argument("--gate-f-denoiser-grad-norm-min", type=float, default=1e-8)
    p.add_argument("--gate-g-zero-loss-tol", type=float, default=1e-8)
    p.add_argument("--gate-g-all-one-match-tol", type=float, default=0.05)
    p.add_argument("--gate-e-steps", type=int, default=6)
    p.add_argument("--gate-e2-min-tps-ratio", type=float, default=0.10)
    p.add_argument("--gate-e2-target-tokens", type=float, default=1e9)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _set_seed(args.seed)

    device = _select_device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"outputs/continuous_embedding/eval/prescale_gates/{_utc_now()}_gates_A_to_E"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _stage(f"output_dir={out_dir}")

    _stage("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    _ensure_special_tokens(tokenizer)

    needed = int(args.microbatch * (args.overfit_batches + args.dev_batches) + 32)
    total_examples = max(needed, args.owt_examples)
    if args.synthetic_data:
        _stage(f"loading synthetic subset: n={total_examples} seq_len={args.seq_len}")
        input_ids, attention_mask = _load_synthetic_tensors(
            seq_len=args.seq_len,
            num_examples=total_examples,
            vocab_size=min(len(tokenizer), int(args.synthetic_vocab_size)),
            seed=args.seed + 17,
        )
    else:
        owt_cache = args.owt_cache_dir if args.owt_cache_dir else None
        _stage(f"loading OWT subset: n={total_examples} seq_len={args.seq_len}")
        input_ids, attention_mask = _load_owt_tensors(
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            num_examples=total_examples,
            cache_dir=owt_cache,
        )
    used = min(input_ids.shape[0], max(needed, args.owt_examples))
    input_ids = input_ids[:used]
    attention_mask = attention_mask[:used]

    overfit_count = int(args.microbatch * args.overfit_batches)
    dev_count = int(args.microbatch * args.dev_batches)
    overfit_ids = input_ids[:overfit_count]
    overfit_attn = attention_mask[:overfit_count]
    dev_ids = input_ids[overfit_count : overfit_count + dev_count]
    dev_attn = attention_mask[overfit_count : overfit_count + dev_count]
    signal_ids = input_ids[overfit_count + dev_count :]
    signal_attn = attention_mask[overfit_count + dev_count :]
    if signal_ids.shape[0] == 0:
        signal_ids = dev_ids
        signal_attn = dev_attn

    overfit_batches = _to_batches(
        overfit_ids,
        overfit_attn,
        batch_size=args.microbatch,
        num_batches=args.overfit_batches,
        num_patterns=args.span_patterns,
    )
    dev_batches = _to_batches(
        dev_ids,
        dev_attn,
        batch_size=args.microbatch,
        num_batches=args.dev_batches,
        num_patterns=args.span_patterns,
    )
    signal_total_batches = max(1, int(signal_ids.shape[0] // max(int(args.microbatch), 1)))
    signal_num_batches = max(
        int(args.dev_batches),
        min(int(args.gate_d_train_batches), int(signal_total_batches)),
    )
    signal_train_batches = _to_batches(
        signal_ids,
        signal_attn,
        batch_size=args.microbatch,
        num_batches=signal_num_batches,
        num_patterns=args.span_patterns,
    )

    cfg = _build_cfg(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        objective=args.objective,
        interpolant=args.interpolant,
        attn_backend=args.attn_backend,
        gradient_checkpointing=False,
        embedding_provider=args.embedding_provider,
        embedding_trainable=bool(args.embedding_trainable),
        lambda_ce=args.lambda_ce,
        self_conditioning=bool(args.self_conditioning),
        self_conditioning_prob=args.self_conditioning_prob,
        min_snr_gamma=args.min_snr_gamma if args.objective == "ddpm" else None,
        imf_alpha=args.imf_alpha,
        imf_warmup_steps=args.imf_warmup_steps,
        imf_ramp_steps=args.imf_ramp_steps,
        train_timesteps=args.train_timesteps,
        vocab_size=len(tokenizer),
    )
    OmegaConf.save(cfg, out_dir / "config_used.yaml")

    _stage("initializing model")
    algo = ContinuousEmbeddingDiffusion(cfg, tokenizer).to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in algo.state_dict().items()}
    jvp_guard = _ensure_jvp_safe_backend(algo)
    _safe_json_dump(jvp_guard, out_dir / "jvp_probe_initial.json")
    dev_baseline_raw = _evaluate_batches(
        algo,
        [
            Batch(
                input_ids=b.input_ids.to(device),
                attention_mask=b.attention_mask.to(device),
                span_mask=b.span_mask.to(device),
            )
            for b in dev_batches
        ],
    )
    dev_baseline_metrics = {
        "masked_ce_mean": float(dev_baseline_raw["ce_mean"]),
        "masked_token_acc_mean": float(dev_baseline_raw["masked_acc_mean"]),
    }
    _safe_json_dump(dev_baseline_metrics, out_dir / "gate_d_baseline_before_training.json")

    recon_t_values = [
        float(x.strip())
        for x in str(args.gate_a_recon_t_values).split(",")
        if x.strip()
    ]
    if not recon_t_values:
        raise ValueError("--gate-a-recon-t-values must contain at least one value")

    gate_a_suite = run_gate_a_suite(
        algo=algo,
        tokenizer=tokenizer,
        cfg=cfg,
        overfit_batches=overfit_batches,
        steps=args.gate_a_steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        autocast_bf16=bool(args.autocast_bf16),
        device=device,
        out_dir=out_dir,
        recon_t_values=recon_t_values,
        recon_low_t=float(args.gate_a_recon_low_t),
        recon_mid_t=float(args.gate_a_recon_mid_t),
        recon_low_acc=float(args.gate_a_recon_low_acc),
        recon_low_ce=float(args.gate_a_recon_low_ce),
        recon_mid_acc=float(args.gate_a_recon_mid_acc),
        recon_high_acc_improve=float(args.gate_a_recon_high_acc_improve),
        recon_high_ce_improve=float(args.gate_a_recon_high_ce_improve),
        overfit_pre_acc_min=float(args.gate_a_overfit_pre_acc_min),
        overfit_post_acc_min=float(args.gate_a_overfit_post_acc_min),
        overfit_fit_acc_min=float(args.gate_a_overfit_fit_acc_min),
        roundtrip_fit_steps=int(args.gate_a_roundtrip_fit_steps),
        roundtrip_fit_lr=float(args.gate_a_roundtrip_fit_lr),
        train_embedding_in_gate_a=bool(args.gate_a_train_embedding),
    )

    gate_b = run_gate_b(
        cfg=cfg,
        tokenizer_name=args.tokenizer_name,
        batch=overfit_batches[0],
        out_dir=out_dir,
        device=device,
        seed=args.seed,
        world_size=args.gate_b_world_size,
        lr=args.lr,
        autocast_bf16=bool(args.autocast_bf16),
        attn_backend=args.attn_backend,
    )
    gate_b_cuda = _evaluate_gate_b_cuda_requirement(
        gate_b=gate_b,
        device=device,
        require_b_cuda=bool(args.require_b_cuda),
        artifact_path=str(args.gate_b_cuda_artifact),
    )
    _safe_json_dump(gate_b_cuda, out_dir / "gate_b_cuda_requirement.json")

    gate_c = run_gate_c(
        algo=algo,
        tokenizer=tokenizer,
        batch=dev_batches[0],
        device=device,
        out_dir=out_dir,
        eps_tolerance=args.gate_c_drift_tol,
    )

    gate_f = run_gate_f(
        algo=algo,
        batch=overfit_batches[0],
        device=device,
        out_dir=out_dir,
        denoiser_grad_norm_min=float(args.gate_f_denoiser_grad_norm_min),
    )

    gate_g = run_gate_g(
        algo=algo,
        batch=overfit_batches[0],
        device=device,
        out_dir=out_dir,
        zero_loss_tol=float(args.gate_g_zero_loss_tol),
        all_one_match_tol=float(args.gate_g_all_one_match_tol),
    )

    gate_d = run_gate_d(
        algo=algo,
        cfg=cfg,
        tokenizer=tokenizer,
        dev_batches=dev_batches,
        train_batches=signal_train_batches,
        device=device,
        out_dir=out_dir,
        baseline_metrics=dev_baseline_metrics,
        signal_ce_drop_ratio_min=float(args.gate_d_signal_ce_drop_ratio_min),
        signal_acc_gain_min=float(args.gate_d_signal_acc_gain_min),
        pilot_steps=int(args.gate_d_pilot_steps),
        pilot_lr=float(args.gate_d_pilot_lr),
        pilot_loss_mode=str(args.gate_d_pilot_loss_mode),
        grad_clip=float(args.grad_clip),
        autocast_bf16=bool(args.autocast_bf16),
        pilot_init_state=initial_state,
    )

    gate_e = run_gate_e(
        cfg_base=cfg,
        tokenizer=tokenizer,
        batch=overfit_batches[0],
        device=device,
        out_dir=out_dir,
        seed=args.seed,
        steps=args.gate_e_steps,
        lr=args.lr,
        autocast_bf16=bool(args.autocast_bf16),
        objective_name=str(args.objective),
        e2_min_tps_ratio=float(args.gate_e2_min_tps_ratio),
        e2_target_tokens=float(args.gate_e2_target_tokens),
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "device": str(device),
            "objective": args.objective,
            "interpolant": args.interpolant,
            "attention_backend": args.attn_backend,
            "owt_examples_used": int(used),
            "synthetic_data": bool(args.synthetic_data),
            "seq_len": int(args.seq_len),
            "microbatch": int(args.microbatch),
            "seed": int(args.seed),
            "gate_d_train_batches": int(args.gate_d_train_batches),
            "gate_d_pilot_steps": int(args.gate_d_pilot_steps),
            "gate_d_pilot_loss_mode": str(args.gate_d_pilot_loss_mode),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "gates": {
            "A_overfit": gate_a_suite["overfit"],
            "A_recon": gate_a_suite["recon"],
            "A_train": gate_a_suite["train"],
            "B": gate_b,
            "B_cuda": gate_b_cuda,
            "C": gate_c,
            "D_stability": gate_d["D_stability"],
            "D_signal": gate_d["D_signal"],
            "F": gate_f,
            "G": gate_g,
            "E": gate_e,
            "E2": gate_e["E2"],
        },
        "gate_details": {
            "A_suite": gate_a_suite,
            "D": gate_d,
        },
    }
    required = ["A_overfit", "A_recon", "C", "D_stability", "F", "G", "E", "E2"]
    if bool(gate_a_suite["recon"]["pass"]) and bool(gate_f["pass"]):
        required.append("D_signal")
    if bool(args.require_b_cuda):
        required.extend(["B", "B_cuda"])
    summary["all_pass"] = bool(all(summary["gates"][g]["pass"] for g in required if g in summary["gates"]))

    _safe_json_dump(summary, out_dir / "summary.json")
    _plot_gate_a_curve(gate_a_suite, out_dir / "gate_a_curve.png")
    _plot_gate_status(summary, out_dir / "gate_status.png")
    _write_summary_markdown(summary, out_dir / "summary.md")

    print(json.dumps({"out_dir": str(out_dir), "all_pass": summary["all_pass"]}, indent=2))
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
