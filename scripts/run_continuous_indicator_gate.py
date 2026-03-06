#!/usr/bin/env python
"""Run an end-to-end continuous-embedding indicator gate on a tiny corpus.

This script validates implementation indicators for:
  - token-id embedding pipeline and legacy compatibility
  - flow matching and iMF/MeanFlow runtime behavior
  - stage-1 CE coupling
  - stabilizers (self-conditioning + Min-SNR path)
  - span-masked infill training signal
  - final external-LM GenPPL thresholds for one-step/few-step generation
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from discrete_diffusion.algorithms.continuous_embedding_diffusion import ContinuousEmbeddingDiffusion
from discrete_diffusion.continuous.diagnostics.jvp_probe import probe_model_jvp
from discrete_diffusion.evaluations.continuous_capabilities.metrics import ExternalPPLEvaluator
from discrete_diffusion.models.common import set_sdpa_math_mode
from discrete_diffusion.sampling.continuous_embedding import ContinuousEmbeddingSampler


class DummyLegacyEncoder(nn.Module):
    """Tiny contextual encoder used for legacy-path compatibility checks."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.output_dim = int(dim)
        self.emb = nn.Embedding(int(vocab_size), int(dim))
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        return self.emb(input_ids)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_cfg(
    *,
    seq_len: int,
    embed_dim: int,
    objective: str,
    interpolant: str,
    conditioning: str,
    embedding_provider: str,
    embedding_trainable: bool,
    lambda_ce: float,
    self_conditioning: bool,
    self_conditioning_prob: float = 0.5,
    min_snr_gamma: float | None = None,
    alpha_imf: float = 1.0,
    alpha_floor: float = 0.0,
    warmup_steps: int = 0,
    ramp_steps: int = 0,
    legacy_encoder: bool = False,
    vocab_size: int | None = None,
) -> OmegaConf:
    cfg = {
        "algo": {
            "stage": 1,
            "sampling_eps": 1e-3,
            "loss_type": "simple",
            "parameterization": "x0",
            "objective": objective,
            "interpolant": interpolant,
            "conditioning": conditioning,
            "embedding_provider": embedding_provider,
            "embedding_trainable": embedding_trainable,
            "train_timesteps": 128,
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
                "alpha_floor": float(alpha_floor),
                "alpha_imf": float(alpha_imf),
                "warmup_steps": int(warmup_steps),
                "ramp_steps": int(ramp_steps),
                "fm_aux_weight": 0.05,
            },
            "span_min_spans": 1,
            "span_max_spans": 3,
            "span_length_dist": "geometric",
            "span_mean_length": 3.0,
            "decoder": {
                "stage2": {
                    "enabled": False,
                }
            },
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
            "hidden_size": int(embed_dim),
            "n_heads": 4,
            "n_blocks": 1,
            "mlp_ratio": 2,
            "dropout": 0.0,
            "time_embed_dim": int(embed_dim),
            "gradient_checkpointing": False,
            "decoder_hidden_size": int(embed_dim),
            "decoder_n_heads": 4,
            "decoder_n_blocks": 1,
            "length": int(seq_len),
            "attn_backend": "sdpa",
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
    }
    if legacy_encoder:
        if vocab_size is None:
            raise ValueError("vocab_size is required when legacy_encoder=True")
        cfg["encoder"] = {
            "_target_": "__main__.DummyLegacyEncoder",
            "vocab_size": int(vocab_size),
            "dim": int(embed_dim),
        }
    return OmegaConf.create(cfg)


def _build_dataset(tokenizer, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = [
        "the company said the market was strong and demand remained high",
        "the united states economy grew as inflation slowed in recent months",
        "the report said the new model improved performance and reliability",
        "investors said the stock market moved higher after the announcement",
        "the team said the new system was tested and ready for release",
        "the data showed that the results were stable across experiments",
        "the government said the policy would support jobs and growth",
        "the article said the method was simple and easy to reproduce",
        "the analysis said the model generated clear and coherent text",
        "the researchers said the approach reduced noise and improved quality",
    ]
    texts = base * 64
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    return tok["input_ids"], tok["attention_mask"]


def _set_fake_trainer(algo, max_steps: int) -> None:
    algo._trainer = SimpleNamespace(global_step=0, max_steps=max_steps, accumulate_grad_batches=1)


def _train_model(
    algo,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    algo.to(device)
    algo.train()
    params = [p for p in algo._get_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    _set_fake_trainer(algo, max_steps=steps)

    losses: list[float] = []
    ce_losses: list[float] = []
    n = input_ids.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        b_ids = input_ids[idx].to(device)
        b_mask = attention_mask[idx].to(device)

        opt.zero_grad(set_to_none=True)
        out = algo._loss(b_ids, b_mask, train_mode=True)
        if not torch.isfinite(out.loss):
            raise RuntimeError(f"non-finite loss at step={step}: {out.loss}")
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        algo._trainer.global_step = step + 1

        losses.append(float(out.loss.detach().cpu().item()))
        ce_losses.append(float(out.ce_loss.detach().cpu().item()))
    return losses, ce_losses


def _generate_texts(
    algo,
    tokenizer,
    *,
    sampling_method: str,
    num_steps: int,
    num_samples: int,
    seq_len: int,
    rectified_init: str,
    rectified_common_token_id: int | None,
) -> list[str]:
    sampler = ContinuousEmbeddingSampler(
        config=algo.config,
        sampling_method=sampling_method,
        eta=0.0,
        temperature=1.0,
        rectified_init=rectified_init,
        rectified_common_token_id=rectified_common_token_id,
    )
    tokens = sampler.generate(
        model=algo,
        num_samples=int(num_samples),
        num_steps=int(num_steps),
        seq_len=int(seq_len),
        return_embeddings=False,
    )
    texts = tokenizer.batch_decode(tokens.detach().cpu(), skip_special_tokens=True)
    return [t.strip() if t.strip() else "the" for t in texts]


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _diversity_metrics(texts: list[str]) -> dict[str, float]:
    unigram_total = 0
    bigram_total = 0
    unigrams: set[str] = set()
    bigrams: set[tuple[str, str]] = set()
    unique_ratios: list[float] = []

    for text in texts:
        toks = _tokenize_words(text)
        if not toks:
            unique_ratios.append(0.0)
            continue
        unigram_total += len(toks)
        unigrams.update(toks)
        if len(toks) > 1:
            pairs = list(zip(toks[:-1], toks[1:]))
            bigram_total += len(pairs)
            bigrams.update(pairs)
        unique_ratios.append(len(set(toks)) / float(len(toks)))

    distinct_1 = float(len(unigrams) / max(unigram_total, 1))
    distinct_2 = float(len(bigrams) / max(bigram_total, 1))
    avg_unique_ratio = float(np.mean(unique_ratios)) if unique_ratios else 0.0
    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "avg_unique_ratio": avg_unique_ratio,
    }


def _printable_ratio(texts: list[str]) -> float:
    ratios: list[float] = []
    for text in texts:
        if not text:
            ratios.append(0.0)
            continue
        printable = sum(1 for ch in text if 32 <= ord(ch) <= 126)
        ratios.append(printable / max(len(text), 1))
    return float(np.mean(ratios))


def _masked_infill_accuracy(
    algo,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    batches: int,
    batch_size: int,
    device: torch.device,
) -> float:
    algo.eval()
    correct = 0
    total = 0
    n = input_ids.shape[0]
    with torch.no_grad():
        for _ in range(batches):
            idx = torch.randint(0, n, (batch_size,))
            b_ids = input_ids[idx].to(device)
            b_mask = attention_mask[idx].to(device)
            span_mask, _ = algo.span_masker.make_batch(
                input_ids=b_ids,
                attention_mask=b_mask,
                mask_token_id=None,
            )
            z0 = algo._embed_inputs(b_ids, b_mask)
            out = algo.objective.compute(
                model=algo,
                x0=z0,
                attention_mask=b_mask,
                span_mask=span_mask,
                input_ids=b_ids,
            )
            logits = algo._decode_logits(out.x0_hat)
            pred = logits.argmax(dim=-1)
            valid = span_mask.bool() & b_mask.bool()
            correct += int((pred[valid] == b_ids[valid]).sum().item())
            total += int(valid.sum().item())
    if total == 0:
        return 0.0
    return float(correct / total)


def _stage_log(msg: str) -> None:
    print(f"[gate] {msg}", flush=True)


def _ensure_jvp_safe_backend(algo: ContinuousEmbeddingDiffusion) -> tuple[bool, str, bool]:
    """Return (ok, message, fallback_applied) after enforcing a JVP-safe attention path."""
    set_sdpa_math_mode(False)
    probe = probe_model_jvp(algo)
    if probe.ok:
        return True, probe.message, False
    if hasattr(algo.denoiser, "set_attention_backend"):
        algo.denoiser.set_attention_backend("sdpa")
    set_sdpa_math_mode(True)
    probe = probe_model_jvp(algo)
    return bool(probe.ok), probe.message, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--steps-fm", type=int, default=140)
    parser.add_argument("--steps-imf", type=int, default=220)
    parser.add_argument("--num-samples", type=int, default=96)
    parser.add_argument("--ppl-model", type=str, default="distilgpt2")
    parser.add_argument("--ppl-batch-size", type=int, default=4)
    parser.add_argument("--ppl-max-length", type=int, default=96)
    parser.add_argument("--rectified-init", type=str, default="random_normal", choices=["random_normal", "fixed_common"])
    parser.add_argument("--rectified-common-token-id", type=int, default=-1)
    parser.add_argument("--rectified-common-token-text", type=str, default=" the")
    parser.add_argument("--dump-samples-path", type=str, default="")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device)

    _stage_log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rectified_common_token_id: int | None
    if int(args.rectified_common_token_id) >= 0:
        rectified_common_token_id = int(args.rectified_common_token_id)
    else:
        ids = tokenizer.encode(str(args.rectified_common_token_text), add_special_tokens=False)
        rectified_common_token_id = int(ids[0]) if ids else int(tokenizer.eos_token_id)

    _stage_log("building tiny corpus")
    input_ids, attention_mask = _build_dataset(tokenizer, seq_len=args.seq_len)

    results: dict[str, object] = {}

    # P1 token-id path
    _stage_log("P1 token-id path check")
    cfg_token = _make_cfg(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        objective="ddpm",
        interpolant="vp",
        conditioning="none",
        embedding_provider="tied",
        embedding_trainable=True,
        lambda_ce=0.2,
        self_conditioning=False,
        min_snr_gamma=5.0,
    )
    algo_token = ContinuousEmbeddingDiffusion(cfg_token, tokenizer).to(device)
    out_token = algo_token._loss(input_ids[:8].to(device), attention_mask[:8].to(device), train_mode=True)
    emb = algo_token._embed_inputs(input_ids[:2].to(device), attention_mask[:2].to(device))
    logits = algo_token._decode_logits(emb)
    results["p1_token_pipeline"] = {
        "ok": bool(torch.isfinite(out_token.loss).item() and logits.shape[:2] == input_ids[:2].shape),
        "loss": float(out_token.loss.item()),
        "logits_shape": list(logits.shape),
    }
    del algo_token
    gc.collect()

    # P1 legacy path
    _stage_log("P1 legacy compatibility path check")
    cfg_legacy = _make_cfg(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        objective="ddpm",
        interpolant="vp",
        conditioning="none",
        embedding_provider="legacy_contextual",
        embedding_trainable=False,
        lambda_ce=0.0,
        self_conditioning=False,
        legacy_encoder=True,
        vocab_size=len(tokenizer),
    )
    algo_legacy = ContinuousEmbeddingDiffusion(cfg_legacy, tokenizer).to(device)
    out_legacy = algo_legacy._loss(input_ids[:8].to(device), attention_mask[:8].to(device), train_mode=True)
    results["p1_legacy_flag"] = {
        "ok": bool(torch.isfinite(out_legacy.loss).item()),
        "loss": float(out_legacy.loss.item()),
    }
    del algo_legacy
    gc.collect()

    # P1bis flow matching
    _stage_log("P1bis flow-matching baseline train/sample")
    cfg_fm = _make_cfg(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        objective="flow_matching",
        interpolant="rectified",
        conditioning="span_masking",
        embedding_provider="tied",
        embedding_trainable=True,
        lambda_ce=0.2,
        self_conditioning=False,
    )
    algo_fm = ContinuousEmbeddingDiffusion(cfg_fm, tokenizer).to(device)
    losses_fm, _ = _train_model(
        algo_fm,
        input_ids,
        attention_mask,
        steps=args.steps_fm,
        batch_size=args.batch_size,
        lr=3e-3,
        device=device,
    )
    fm_one = _generate_texts(
        algo_fm,
        tokenizer,
        sampling_method="rectified_one_step",
        num_steps=1,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        rectified_init=args.rectified_init,
        rectified_common_token_id=rectified_common_token_id,
    )
    fm_printable = _printable_ratio(fm_one)
    results["p1bis_flow_matching"] = {
        "ok": bool(losses_fm[-1] < losses_fm[0] and fm_printable > 0.95),
        "loss_start": float(losses_fm[0]),
        "loss_end": float(losses_fm[-1]),
        "one_step_printable_ratio": float(fm_printable),
    }
    del algo_fm
    gc.collect()

    # P1bis iMF + P4 infill
    _stage_log("P1bis iMF/MeanFlow train + P4 infill progression")
    cfg_imf = _make_cfg(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        objective="meanflow",
        interpolant="rectified",
        conditioning="span_masking",
        embedding_provider="tied",
        embedding_trainable=True,
        lambda_ce=0.2,
        self_conditioning=True,
        alpha_imf=1.0,
        alpha_floor=0.0,
        warmup_steps=40,
        ramp_steps=80,
    )
    algo_imf = ContinuousEmbeddingDiffusion(cfg_imf, tokenizer).to(device)
    probe_ok, probe_msg, fallback_applied = _ensure_jvp_safe_backend(algo_imf)

    pre_infill_acc = _masked_infill_accuracy(
        algo_imf,
        input_ids,
        attention_mask,
        batches=6,
        batch_size=min(16, args.batch_size),
        device=device,
    )
    losses_imf, ce_losses_imf = _train_model(
        algo_imf,
        input_ids,
        attention_mask,
        steps=args.steps_imf,
        batch_size=args.batch_size,
        lr=3e-3,
        device=device,
    )
    post_infill_acc = _masked_infill_accuracy(
        algo_imf,
        input_ids,
        attention_mask,
        batches=6,
        batch_size=min(16, args.batch_size),
        device=device,
    )

    _set_fake_trainer(algo_imf, max_steps=args.steps_imf)
    algo_imf._trainer.global_step = 0
    alpha_start = float(algo_imf._current_imf_alpha())
    algo_imf._trainer.global_step = args.steps_imf
    alpha_end = float(algo_imf._current_imf_alpha())

    one_step_texts = _generate_texts(
        algo_imf,
        tokenizer,
        sampling_method="rectified_one_step",
        num_steps=1,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        rectified_init=args.rectified_init,
        rectified_common_token_id=rectified_common_token_id,
    )

    few_step_texts = _generate_texts(
        algo_imf,
        tokenizer,
        sampling_method="rectified_few_step",
        num_steps=8,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        rectified_init=args.rectified_init,
        rectified_common_token_id=rectified_common_token_id,
    )

    if args.dump_samples_path:
        dump_path = Path(args.dump_samples_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            json.dumps(
                {
                    "one_step_texts": one_step_texts,
                    "few_step_texts": few_step_texts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    results["p1bis_imf"] = {
        "ok": bool(probe_ok and math.isfinite(losses_imf[-1])),
        "jvp_probe_ok": bool(probe_ok),
        "jvp_probe_message": probe_msg,
        "jvp_fallback_applied": bool(fallback_applied),
        "rectified_init": str(args.rectified_init),
        "rectified_common_token_id": int(rectified_common_token_id) if rectified_common_token_id is not None else None,
        "loss_start": float(losses_imf[0]),
        "loss_end": float(losses_imf[-1]),
        "ce_loss_end": float(ce_losses_imf[-1]),
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
    }

    # P2 stage-1 CE coupling
    _stage_log("P2 stage-1 CE coupling check")
    out_stage1 = algo_imf._loss(input_ids[:16].to(device), attention_mask[:16].to(device), train_mode=True)
    results["p2_joint_ce"] = {
        "ok": bool(out_stage1.ce_loss is not None and float(out_stage1.ce_loss.item()) > 0.0 and not cfg_imf.algo.decoder.stage2.enabled),
        "ce_loss": float(out_stage1.ce_loss.item()),
        "stage2_enabled": bool(cfg_imf.algo.decoder.stage2.enabled),
    }

    # P3 stabilizers
    _stage_log("P3 stabilizer checks (self-conditioning ablation + Min-SNR path)")
    cfg_sc_off = _make_cfg(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        objective="meanflow",
        interpolant="rectified",
        conditioning="span_masking",
        embedding_provider="tied",
        embedding_trainable=True,
        lambda_ce=0.2,
        self_conditioning=False,
    )
    algo_sc_off = ContinuousEmbeddingDiffusion(cfg_sc_off, tokenizer).to(device)
    _ensure_jvp_safe_backend(algo_sc_off)
    loss_sc_off = algo_sc_off._loss(input_ids[:8].to(device), attention_mask[:8].to(device), train_mode=True).loss
    del algo_sc_off

    cfg_sc_on = _make_cfg(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        objective="meanflow",
        interpolant="rectified",
        conditioning="span_masking",
        embedding_provider="tied",
        embedding_trainable=True,
        lambda_ce=0.2,
        self_conditioning=True,
    )
    algo_sc_on = ContinuousEmbeddingDiffusion(cfg_sc_on, tokenizer).to(device)
    _ensure_jvp_safe_backend(algo_sc_on)
    loss_sc_on = algo_sc_on._loss(input_ids[:8].to(device), attention_mask[:8].to(device), train_mode=True).loss
    del algo_sc_on

    cfg_minsnr = _make_cfg(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        objective="ddpm",
        interpolant="vp",
        conditioning="none",
        embedding_provider="tied",
        embedding_trainable=False,
        lambda_ce=0.0,
        self_conditioning=False,
        min_snr_gamma=5.0,
    )
    algo_minsnr = ContinuousEmbeddingDiffusion(cfg_minsnr, tokenizer).to(device)
    loss_minsnr = algo_minsnr._loss(input_ids[:8].to(device), attention_mask[:8].to(device), train_mode=True).loss
    del algo_minsnr
    gc.collect()

    results["p3_stabilizers"] = {
        "ok": bool(torch.isfinite(loss_sc_off).item() and torch.isfinite(loss_sc_on).item() and torch.isfinite(loss_minsnr).item()),
        "self_cond_off_loss": float(loss_sc_off.item()),
        "self_cond_on_loss": float(loss_sc_on.item()),
        "ddpm_min_snr_loss": float(loss_minsnr.item()),
    }

    # P4 infill
    results["p4_infill"] = {
        "ok": bool(post_infill_acc > (pre_infill_acc + 0.01) and post_infill_acc >= 0.02),
        "pre_masked_token_acc": float(pre_infill_acc),
        "post_masked_token_acc": float(post_infill_acc),
    }

    # Release training models before loading external LM for PPL to reduce memory pressure.
    del algo_imf
    gc.collect()

    # Final external GenPPL gate
    _stage_log("final GenPPL gate")
    ppl_eval = ExternalPPLEvaluator(
        args.ppl_model,
        batch_size=max(1, int(args.ppl_batch_size)),
        max_length=max(16, int(args.ppl_max_length)),
        device=device,
    )
    ppl_one = ppl_eval.evaluate(one_step_texts)
    ppl_few = ppl_eval.evaluate(few_step_texts)
    diversity_one = _diversity_metrics(one_step_texts)
    diversity_few = _diversity_metrics(few_step_texts)
    one_ppl = float(ppl_one.get("ppl", float("nan")))
    few_ppl = float(ppl_few.get("ppl", float("nan")))
    quality_ok = bool(
        diversity_one["distinct_2"] >= 0.02
        and diversity_few["distinct_2"] >= 0.02
        and diversity_one["avg_unique_ratio"] >= 0.25
        and diversity_few["avg_unique_ratio"] >= 0.25
    )
    results["final_genppl"] = {
        "one_step_ppl": one_ppl,
        "few_step_ppl": few_ppl,
        "one_step_pass": bool(np.isfinite(one_ppl) and one_ppl < 150.0),
        "few_step_pass": bool(np.isfinite(few_ppl) and few_ppl < 100.0),
        "proxy_pass": bool(np.isfinite(one_ppl) and one_ppl < 150.0 and np.isfinite(few_ppl) and few_ppl < 100.0),
        "quality_pass": quality_ok,
        "one_step_diversity": diversity_one,
        "few_step_diversity": diversity_few,
        "lm_available": bool(ppl_one.get("available", False) and ppl_few.get("available", False)),
        "lm_error": ppl_one.get("error", None) or ppl_few.get("error", None),
    }

    results["samples"] = {
        "one_step_head": one_step_texts[:5],
        "few_step_head": few_step_texts[:5],
    }

    all_green = (
        results["p1_token_pipeline"]["ok"]
        and results["p1_legacy_flag"]["ok"]
        and results["p1bis_flow_matching"]["ok"]
        and results["p1bis_imf"]["ok"]
        and results["p2_joint_ce"]["ok"]
        and results["p3_stabilizers"]["ok"]
        and results["p4_infill"]["ok"]
        and results["final_genppl"]["proxy_pass"]
    )
    results["all_green"] = bool(all_green)

    print(json.dumps(results, indent=2))
    return 0 if all_green else 2


if __name__ == "__main__":
    raise SystemExit(main())
