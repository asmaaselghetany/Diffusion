"""Task runners for continuous capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import QAPromptRecord, choose_source_subset
from .metrics import (
    ExternalPPLEvaluator,
    MetricAccumulator,
    compute_diversity_metrics,
    compute_prompt_qa_metrics,
    finalize_reconstruction_metrics,
    update_reconstruction_accumulator,
)
from .sampling import decode_embeddings, sample_latents_conditioned, sample_latents_unconditional


@dataclass
class SamplingConfig:
    """Sampling options shared across tasks."""

    num_steps: int
    sampling_method: str
    eta: float
    eps: float
    temperature: float
    top_p: float
    top_k: int
    ban_special_tokens: bool
    clip_denoised: bool
    clip_range: Tuple[float, float]
    adaptive_lm_guidance_scale: float = 0.0
    adaptive_lm_model_name: str = "gpt2"
    adaptive_lm_guidance_mode: str = "confidence"
    adaptive_lm_guidance_min_scale: float = 0.0
    adaptive_lm_guidance_scale_conditioned: Optional[float] = None
    max_decode_tokens: Optional[int] = None


_LM_GUIDANCE_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _get_lm_guidance_model(
    model,
    *,
    sampling_cfg: SamplingConfig,
):
    lm_guidance_scale = max(0.0, float(getattr(sampling_cfg, "adaptive_lm_guidance_scale", 0.0)))
    if lm_guidance_scale <= 0:
        return None, lm_guidance_scale

    from transformers import AutoModelForCausalLM

    lm_name = str(getattr(sampling_cfg, "adaptive_lm_model_name", "gpt2"))
    model_device = next(model.denoiser.parameters()).device
    cache_key = (lm_name, str(model_device))
    lm_guidance_model = _LM_GUIDANCE_MODEL_CACHE.get(cache_key)
    if lm_guidance_model is None:
        lm_guidance_model = AutoModelForCausalLM.from_pretrained(lm_name)
        lm_guidance_model.to(model_device)
        lm_guidance_model.eval()
        _LM_GUIDANCE_MODEL_CACHE[cache_key] = lm_guidance_model

    return lm_guidance_model, lm_guidance_scale


def _conditioned_sampling_cfg(sampling_cfg: SamplingConfig) -> SamplingConfig:
    conditioned_scale = getattr(sampling_cfg, "adaptive_lm_guidance_scale_conditioned", None)
    if conditioned_scale is None:
        return sampling_cfg
    conditioned_scale = float(conditioned_scale)
    conditioned_min_scale = min(
        float(getattr(sampling_cfg, "adaptive_lm_guidance_min_scale", 0.0)),
        conditioned_scale,
    )
    return replace(
        sampling_cfg,
        adaptive_lm_guidance_scale=conditioned_scale,
        adaptive_lm_guidance_min_scale=conditioned_min_scale,
    )



def _merge_accumulators(dst: MetricAccumulator, src: MetricAccumulator) -> None:
    dst.matched_tokens += src.matched_tokens
    dst.total_tokens += src.total_tokens
    dst.exact_matches += src.exact_matches
    dst.total_sequences += src.total_sequences
    dst.ce_sum += src.ce_sum



def _iter_tensor_batches(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
):
    for start in range(0, int(input_ids.shape[0]), batch_size):
        end = min(start + batch_size, int(input_ids.shape[0]))
        yield start, end, input_ids[start:end], attention_mask[start:end]



def _decode_texts(tokenizer, token_ids: torch.Tensor) -> List[str]:
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)



def run_reconstruction_encode_decode(
    model,
    tokenizer,
    *,
    sources: Sequence[Dict[str, Any]],
    batch_size: int,
    save_samples: bool,
    sample_save_limit: int,
) -> Dict[str, Any]:
    """Evaluate pure encoder->decoder reconstruction (no denoiser)."""
    out: Dict[str, Any] = {
        "status": "ok",
        "sources": {},
        "overall": {},
    }

    overall_acc = MetricAccumulator()
    saved_samples: List[Dict[str, Any]] = []

    with torch.no_grad():
        for source in sources:
            source_name = source["source"]
            input_ids = source["input_ids"]
            attention_mask = source["attention_mask"]
            source_acc = MetricAccumulator()

            for start, end, batch_ids, batch_attn in _iter_tensor_batches(input_ids, attention_mask, batch_size):
                z_0 = model.encoder(batch_ids, batch_attn)
                logits = model.decoder(z_0)
                pred_ids = logits.argmax(dim=-1)

                token_mask = batch_attn.bool()
                update_reconstruction_accumulator(
                    source_acc,
                    target_ids=batch_ids,
                    pred_ids=pred_ids,
                    logits=logits,
                    token_mask=token_mask,
                )

                if save_samples and len(saved_samples) < sample_save_limit:
                    decoded_inputs = _decode_texts(tokenizer, batch_ids)
                    decoded_preds = _decode_texts(tokenizer, pred_ids)
                    sample_ids = source.get("sample_ids")
                    for local_idx in range(batch_ids.shape[0]):
                        if len(saved_samples) >= sample_save_limit:
                            break
                        global_idx = start + local_idx
                        saved_samples.append(
                            {
                                "task": "reconstruction",
                                "source": source_name,
                                "sample_id": sample_ids[global_idx] if sample_ids else f"{source_name}_{global_idx}",
                                "input_text": decoded_inputs[local_idx],
                                "prediction_text": decoded_preds[local_idx],
                            }
                        )

            source_metrics = finalize_reconstruction_metrics(source_acc)
            out["sources"][source_name] = source_metrics
            _merge_accumulators(overall_acc, source_acc)

    out["overall"] = finalize_reconstruction_metrics(overall_acc)
    if save_samples:
        out["samples"] = saved_samples
    return out



def run_unconditional_generation(
    model,
    tokenizer,
    *,
    num_samples: int,
    generation_batch_size: int,
    sampling_cfg: SamplingConfig,
    ppl_evaluator: Optional[ExternalPPLEvaluator],
    save_samples: bool,
    sample_save_limit: int,
) -> Dict[str, Any]:
    """Evaluate unconditional generation from Gaussian latent noise."""
    seq_len = int(model.config.model.length)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    additional_banned = [int(mask_id)] if mask_id is not None else None
    generated_ids: List[torch.Tensor] = []
    lm_guidance_model, lm_guidance_scale = _get_lm_guidance_model(model, sampling_cfg=sampling_cfg)
    lm_guidance_start_token_id = int(getattr(tokenizer, "eos_token_id", 0) or 0)

    remaining = num_samples
    with torch.no_grad():
        while remaining > 0:
            batch_n = min(remaining, generation_batch_size)
            z_t = sample_latents_unconditional(
                model,
                num_samples=batch_n,
                seq_len=seq_len,
                num_steps=sampling_cfg.num_steps,
                sampling_method=sampling_cfg.sampling_method,  # type: ignore[arg-type]
                eta=sampling_cfg.eta,
                eps=sampling_cfg.eps,
                clip_denoised=sampling_cfg.clip_denoised,
                clip_range=sampling_cfg.clip_range,
            )
            token_ids, _ = decode_embeddings(
                model,
                tokenizer,
                z_t,
                temperature=sampling_cfg.temperature,
                top_p=sampling_cfg.top_p,
                top_k=sampling_cfg.top_k,
                ban_special_tokens=sampling_cfg.ban_special_tokens,
                additional_banned_token_ids=additional_banned,
                lm_guidance_model=lm_guidance_model,
                lm_guidance_scale=lm_guidance_scale,
                lm_guidance_start_token_id=lm_guidance_start_token_id,
                lm_guidance_mode=str(getattr(sampling_cfg, "adaptive_lm_guidance_mode", "confidence")),
                lm_guidance_min_scale=float(getattr(sampling_cfg, "adaptive_lm_guidance_min_scale", 0.0)),
            )
            if sampling_cfg.max_decode_tokens is not None and sampling_cfg.max_decode_tokens > 0:
                token_ids = token_ids[:, : int(min(token_ids.shape[1], sampling_cfg.max_decode_tokens))]
            generated_ids.append(token_ids.cpu())
            remaining -= batch_n

    all_ids = torch.cat(generated_ids, dim=0)
    texts = _decode_texts(tokenizer, all_ids)

    metrics = {
        "status": "ok",
        "num_samples": int(all_ids.shape[0]),
        "diversity": compute_diversity_metrics(all_ids, texts),
    }
    if lm_guidance_model is not None:
        metrics["adaptive_sampling"] = {
            "lm_guidance_model": str(getattr(sampling_cfg, "adaptive_lm_model_name", "gpt2")),
            "lm_guidance_scale": lm_guidance_scale,
            "lm_guidance_mode": str(getattr(sampling_cfg, "adaptive_lm_guidance_mode", "confidence")),
            "lm_guidance_min_scale": float(getattr(sampling_cfg, "adaptive_lm_guidance_min_scale", 0.0)),
            "max_decode_tokens": sampling_cfg.max_decode_tokens,
        }

    if ppl_evaluator is not None:
        metrics["external_ppl"] = ppl_evaluator.evaluate(texts)

    if save_samples:
        metrics["samples"] = [
            {
                "task": "unconditional",
                "sample_id": f"gen_{idx}",
                "prediction_text": text,
            }
            for idx, text in enumerate(texts[:sample_save_limit])
        ]

    return metrics



def _sample_span_mask(
    attention_mask: torch.Tensor,
    *,
    rng: np.random.Generator,
    span_fraction: float,
    min_span_tokens: int,
) -> torch.Tensor:
    """Create one contiguous infill span per sequence."""
    batch_size, seq_len = attention_mask.shape
    span_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

    for row in range(batch_size):
        valid_len = int(attention_mask[row].sum().item())
        if valid_len <= 1:
            continue

        span_len = max(min_span_tokens, int(round(valid_len * span_fraction)))
        span_len = min(span_len, valid_len)

        max_start = max(0, valid_len - span_len)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end = min(valid_len, start + span_len)
        span_mask[row, start:end] = True

    return span_mask



def run_infilling(
    model,
    tokenizer,
    *,
    sources: Sequence[Dict[str, Any]],
    num_samples: int,
    generation_batch_size: int,
    sampling_cfg: SamplingConfig,
    span_fraction: float,
    min_span_tokens: int,
    seed: int,
    save_samples: bool,
    sample_save_limit: int,
) -> Dict[str, Any]:
    """Evaluate infilling with contiguous masked spans."""
    rng = np.random.default_rng(seed)
    mask_id = int(getattr(tokenizer, "mask_token_id", 0))

    out: Dict[str, Any] = {"status": "ok", "sources": {}, "overall": {}}
    overall_acc = MetricAccumulator()
    saved_samples: List[Dict[str, Any]] = []
    conditioned_sampling_cfg = _conditioned_sampling_cfg(sampling_cfg)
    lm_guidance_model, lm_guidance_scale = _get_lm_guidance_model(
        model,
        sampling_cfg=conditioned_sampling_cfg,
    )
    lm_guidance_start_token_id = int(getattr(tokenizer, "eos_token_id", 0) or 0)

    if not sources:
        out["status"] = "skipped"
        out["reason"] = "no sources available"
        return out

    per_source_budget = max(1, num_samples // len(sources))

    with torch.no_grad():
        for source_idx, source in enumerate(sources):
            source_name = source["source"]
            subset_seed = seed + 17 * (source_idx + 1)
            subset = choose_source_subset(source, num_samples=per_source_budget, seed=subset_seed)

            input_ids = subset["input_ids"]
            attention_mask = subset["attention_mask"]
            sample_ids = subset.get("sample_ids")
            source_acc = MetricAccumulator()

            for start, end, batch_ids, batch_attn in _iter_tensor_batches(input_ids, attention_mask, generation_batch_size):
                span_mask = _sample_span_mask(
                    batch_attn,
                    rng=rng,
                    span_fraction=span_fraction,
                    min_span_tokens=min_span_tokens,
                )

                conditioning_ids = batch_ids.clone()
                conditioning_ids[span_mask] = mask_id

                fixed_mask = (~span_mask & batch_attn.bool()) | (~batch_attn.bool())

                z_t = sample_latents_conditioned(
                    model,
                    conditioning_ids=conditioning_ids,
                    fixed_mask=fixed_mask,
                    attention_mask=batch_attn,
                    num_steps=sampling_cfg.num_steps,
                    sampling_method=sampling_cfg.sampling_method,  # type: ignore[arg-type]
                    eta=sampling_cfg.eta,
                    eps=sampling_cfg.eps,
                    clip_denoised=sampling_cfg.clip_denoised,
                    clip_range=sampling_cfg.clip_range,
                )

                pred_ids, logits = decode_embeddings(
                    model,
                    tokenizer,
                    z_t,
                    temperature=sampling_cfg.temperature,
                    top_p=sampling_cfg.top_p,
                    top_k=sampling_cfg.top_k,
                    ban_special_tokens=sampling_cfg.ban_special_tokens,
                    additional_banned_token_ids=[mask_id],
                    lm_guidance_model=lm_guidance_model,
                    lm_guidance_scale=lm_guidance_scale,
                    lm_guidance_start_token_id=lm_guidance_start_token_id,
                    lm_guidance_mode=str(getattr(conditioned_sampling_cfg, "adaptive_lm_guidance_mode", "confidence")),
                    lm_guidance_min_scale=float(
                        getattr(conditioned_sampling_cfg, "adaptive_lm_guidance_min_scale", 0.0)
                    ),
                )

                update_reconstruction_accumulator(
                    source_acc,
                    target_ids=batch_ids,
                    pred_ids=pred_ids,
                    logits=logits,
                    token_mask=span_mask,
                )

                if save_samples and len(saved_samples) < sample_save_limit:
                    decoded_input = _decode_texts(tokenizer, batch_ids)
                    decoded_masked = _decode_texts(tokenizer, conditioning_ids)
                    decoded_pred = _decode_texts(tokenizer, pred_ids)
                    for local_idx in range(batch_ids.shape[0]):
                        if len(saved_samples) >= sample_save_limit:
                            break
                        global_idx = start + local_idx
                        saved_samples.append(
                            {
                                "task": "infill",
                                "source": source_name,
                                "sample_id": sample_ids[global_idx] if sample_ids else f"{source_name}_{global_idx}",
                                "input_text": decoded_input[local_idx],
                                "masked_text": decoded_masked[local_idx],
                                "prediction_text": decoded_pred[local_idx],
                            }
                        )

            source_metrics = finalize_reconstruction_metrics(source_acc)
            out["sources"][source_name] = source_metrics
            _merge_accumulators(overall_acc, source_acc)

    out["overall"] = finalize_reconstruction_metrics(overall_acc)
    if save_samples:
        out["samples"] = saved_samples
    return out



def _truncate_on_stop_strings(text: str, stop_strings: Optional[Sequence[str]]) -> str:
    if not stop_strings:
        return text
    min_idx = None
    for stop in stop_strings:
        idx = text.find(stop)
        if idx >= 0:
            min_idx = idx if min_idx is None else min(min_idx, idx)
    if min_idx is None:
        return text
    return text[:min_idx]



def run_prompt_qa(
    model,
    tokenizer,
    *,
    qa_records: Sequence[QAPromptRecord],
    sampling_cfg: SamplingConfig,
    default_max_answer_tokens: int,
    save_samples: bool,
    sample_save_limit: int,
) -> Dict[str, Any]:
    """Evaluate prompt-answering using local JSONL prompts."""
    if not qa_records:
        return {"status": "skipped", "reason": "no QA records provided"}

    model_len = int(model.config.model.length)
    mask_id = int(getattr(tokenizer, "mask_token_id", 0))
    device = next(model.denoiser.parameters()).device
    conditioned_sampling_cfg = _conditioned_sampling_cfg(sampling_cfg)
    lm_guidance_model, lm_guidance_scale = _get_lm_guidance_model(
        model,
        sampling_cfg=conditioned_sampling_cfg,
    )
    lm_guidance_start_token_id = int(getattr(tokenizer, "eos_token_id", 0) or 0)

    per_example: List[Dict[str, str]] = []
    artifacts: List[Dict[str, str]] = []

    with torch.no_grad():
        for rec in qa_records:
            prompt_ids = tokenizer.encode(rec.prompt, add_special_tokens=False)
            suffix_ids = tokenizer.encode(rec.suffix, add_special_tokens=False) if rec.suffix else []

            max_answer = rec.max_answer_tokens or default_max_answer_tokens

            available_for_prompt_and_answer = model_len - len(suffix_ids)
            if available_for_prompt_and_answer <= 1:
                continue

            # Preserve suffix, trim prompt from the left when needed.
            max_prompt_len = max(1, available_for_prompt_and_answer - 1)
            prompt_ids = prompt_ids[-max_prompt_len:]

            answer_len = min(max_answer, model_len - len(prompt_ids) - len(suffix_ids))
            if answer_len <= 0:
                continue

            total_len = len(prompt_ids) + answer_len + len(suffix_ids)
            conditioning_ids = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
            attention_mask = torch.ones_like(conditioning_ids, dtype=torch.long, device=device)
            fixed_mask = torch.zeros_like(conditioning_ids, dtype=torch.bool, device=device)

            prompt_len = len(prompt_ids)
            if prompt_len > 0:
                prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
                conditioning_ids[0, :prompt_len] = prompt_tensor
                fixed_mask[0, :prompt_len] = True

            if suffix_ids:
                suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=device)
                suffix_start = total_len - len(suffix_ids)
                conditioning_ids[0, suffix_start:] = suffix_tensor
                fixed_mask[0, suffix_start:] = True

            z_t = sample_latents_conditioned(
                model,
                conditioning_ids=conditioning_ids,
                fixed_mask=fixed_mask,
                attention_mask=attention_mask,
                num_steps=sampling_cfg.num_steps,
                sampling_method=sampling_cfg.sampling_method,  # type: ignore[arg-type]
                eta=sampling_cfg.eta,
                eps=sampling_cfg.eps,
                clip_denoised=sampling_cfg.clip_denoised,
                clip_range=sampling_cfg.clip_range,
            )

            pred_ids, _ = decode_embeddings(
                model,
                tokenizer,
                z_t,
                temperature=sampling_cfg.temperature,
                top_p=sampling_cfg.top_p,
                top_k=sampling_cfg.top_k,
                ban_special_tokens=sampling_cfg.ban_special_tokens,
                additional_banned_token_ids=[mask_id],
                lm_guidance_model=lm_guidance_model,
                lm_guidance_scale=lm_guidance_scale,
                lm_guidance_start_token_id=lm_guidance_start_token_id,
                lm_guidance_mode=str(getattr(conditioned_sampling_cfg, "adaptive_lm_guidance_mode", "confidence")),
                lm_guidance_min_scale=float(getattr(conditioned_sampling_cfg, "adaptive_lm_guidance_min_scale", 0.0)),
            )

            answer_start = prompt_len
            answer_end = prompt_len + answer_len
            answer_ids = pred_ids[0, answer_start:answer_end]
            predicted_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            predicted_text = _truncate_on_stop_strings(predicted_text, rec.stop_strings)

            per_example.append(
                {
                    "id": rec.id,
                    "prompt": rec.prompt,
                    "answer": rec.answer,
                    "prediction": predicted_text,
                }
            )

            if save_samples and len(artifacts) < sample_save_limit:
                artifacts.append(
                    {
                        "task": "prompt_qa",
                        "sample_id": rec.id,
                        "prompt": rec.prompt,
                        "answer": rec.answer,
                        "prediction": predicted_text,
                    }
                )

    metrics = compute_prompt_qa_metrics(per_example)
    out: Dict[str, Any] = {
        "status": "ok",
        "overall": metrics,
        "count": len(per_example),
    }
    if lm_guidance_model is not None:
        out["adaptive_sampling"] = {
            "lm_guidance_model": str(getattr(sampling_cfg, "adaptive_lm_model_name", "gpt2")),
            "lm_guidance_scale": lm_guidance_scale,
            "lm_guidance_mode": str(getattr(conditioned_sampling_cfg, "adaptive_lm_guidance_mode", "confidence")),
            "lm_guidance_min_scale": float(getattr(conditioned_sampling_cfg, "adaptive_lm_guidance_min_scale", 0.0)),
        }

    if save_samples:
        out["samples"] = artifacts
    return out


__all__ = [
    "SamplingConfig",
    "run_reconstruction_encode_decode",
    "run_unconditional_generation",
    "run_infilling",
    "run_prompt_qa",
]
