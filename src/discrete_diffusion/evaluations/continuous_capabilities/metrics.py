"""Metric helpers for continuous capability evaluation."""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MetricAccumulator:
    """Running accumulator for masked token metrics."""

    matched_tokens: int = 0
    total_tokens: int = 0
    exact_matches: int = 0
    total_sequences: int = 0
    ce_sum: float = 0.0



def _safe_exp(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return float(math.exp(min(value, 80.0)))



def update_reconstruction_accumulator(
    acc: MetricAccumulator,
    *,
    target_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    logits: torch.Tensor,
    token_mask: torch.Tensor,
) -> None:
    """Update a reconstruction accumulator with one batch."""
    valid_mask = token_mask.bool()
    if valid_mask.ndim != 2:
        raise ValueError("token_mask must be a 2D tensor [B, L]")

    matches = (target_ids == pred_ids) & valid_mask
    batch_total = int(valid_mask.sum().item())
    batch_matches = int(matches.sum().item())

    flat_logits = logits.view(-1, logits.shape[-1])
    flat_targets = target_ids.view(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").view_as(target_ids)
    batch_ce = float((ce * valid_mask).sum().item())

    # Exact sequence match over the selected mask only.
    seq_exact = (matches | ~valid_mask).all(dim=-1)

    acc.matched_tokens += batch_matches
    acc.total_tokens += batch_total
    acc.exact_matches += int(seq_exact.sum().item())
    acc.total_sequences += int(target_ids.shape[0])
    acc.ce_sum += batch_ce



def finalize_reconstruction_metrics(acc: MetricAccumulator) -> Dict[str, float]:
    """Finalize token-level reconstruction metrics."""
    if acc.total_tokens <= 0 or acc.total_sequences <= 0:
        return {
            "token_accuracy": float("nan"),
            "sequence_exact_match": float("nan"),
            "avg_ce": float("nan"),
            "intrinsic_ppl": float("nan"),
            "total_tokens": 0,
            "total_sequences": 0,
        }

    avg_ce = acc.ce_sum / max(acc.total_tokens, 1)
    return {
        "token_accuracy": acc.matched_tokens / max(acc.total_tokens, 1),
        "sequence_exact_match": acc.exact_matches / max(acc.total_sequences, 1),
        "avg_ce": avg_ce,
        "intrinsic_ppl": _safe_exp(avg_ce),
        "total_tokens": acc.total_tokens,
        "total_sequences": acc.total_sequences,
    }



def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]



def compute_diversity_metrics(token_ids: torch.Tensor, texts: Sequence[str]) -> Dict[str, float]:
    """Compute diversity and entropy metrics for generated text."""
    unique_text_ratio = len(set(texts)) / max(len(texts), 1)

    corpus_tokens = [tok for text in texts for tok in text.split()]
    uni = _ngrams(corpus_tokens, 1)
    bi = _ngrams(corpus_tokens, 2)

    distinct_1 = len(set(uni)) / max(len(uni), 1)
    distinct_2 = len(set(bi)) / max(len(bi), 1)

    entropies: List[float] = []
    for sample in token_ids:
        _, counts = torch.unique(sample, return_counts=True, sorted=False)
        probs = counts.float() / counts.sum().clamp(min=1)
        entropies.append(float(torch.special.entr(probs).sum().item()))

    return {
        "unique_text_ratio": float(unique_text_ratio),
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "mean_token_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "std_token_entropy": float(np.std(entropies)) if entropies else float("nan"),
        "avg_text_chars": float(np.mean([len(t) for t in texts])) if texts else float("nan"),
    }



def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())



def normalize_answer(text: str) -> str:
    """Normalize answers for robust EM/F1 (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return _normalize_whitespace(text)



def normalized_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))



def normalized_token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}

    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)



def compute_prompt_qa_metrics(entries: Sequence[Dict[str, str]]) -> Dict[str, float]:
    """Compute aggregate prompt-answering metrics from per-example entries."""
    if not entries:
        return {
            "normalized_exact_match": float("nan"),
            "normalized_token_f1": float("nan"),
            "count": 0,
        }

    em_scores = [
        normalized_exact_match(item["prediction"], item["answer"]) for item in entries
    ]
    f1_scores = [
        normalized_token_f1(item["prediction"], item["answer"]) for item in entries
    ]

    return {
        "normalized_exact_match": float(np.mean(em_scores)),
        "normalized_token_f1": float(np.mean(f1_scores)),
        "count": len(entries),
    }


class ExternalPPLEvaluator:
    """Lazy external-LM perplexity evaluator with graceful fallback."""

    def __init__(self, model_name: str, batch_size: int, max_length: int, device: torch.device):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_kwargs = {}
            if self.device.type == "cuda":
                model_kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            if self.device.type != "cuda":
                self._model.to(self.device)
            self._model.eval()
        except Exception as exc:  # pylint: disable=broad-except
            self._load_error = str(exc)
            self._model = None
            self._tokenizer = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None and self._tokenizer is not None

    @property
    def error(self) -> Optional[str]:
        self._ensure_loaded()
        return self._load_error

    def evaluate(self, texts: Sequence[str]) -> Dict[str, float | int | bool | str | None]:
        self._ensure_loaded()
        if self._model is None or self._tokenizer is None:
            return {
                "available": False,
                "error": self._load_error,
                "avg_nll": float("nan"),
                "ppl": float("nan"),
                "median_nll": float("nan"),
                "tokens_evaluated": 0,
            }

        model_device = next(self._model.parameters()).device

        total_nll = 0.0
        total_tokens = 0
        all_nlls: List[float] = []

        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = list(texts[start : start + self.batch_size])
                tokenized = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                )

                input_ids = tokenized["input_ids"].to(model_device)
                attention_mask = tokenized["attention_mask"].to(model_device)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                logits = outputs.logits[:, :-1]
                labels = input_ids[:, 1:]
                loss_mask = attention_mask[:, :-1]

                nll = F.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(0, 1),
                    reduction="none",
                ).view_as(labels)

                valid = loss_mask.bool()
                valid_nll = nll[valid]
                all_nlls.extend(valid_nll.detach().cpu().tolist())

                total_nll += float((nll * loss_mask).sum().item())
                total_tokens += int(valid.sum().item())

        if total_tokens == 0:
            return {
                "available": True,
                "error": None,
                "avg_nll": float("nan"),
                "ppl": float("nan"),
                "median_nll": float("nan"),
                "tokens_evaluated": 0,
            }

        avg_nll = total_nll / total_tokens
        return {
            "available": True,
            "error": None,
            "avg_nll": float(avg_nll),
            "ppl": _safe_exp(avg_nll),
            "median_nll": float(np.median(all_nlls)) if all_nlls else float("nan"),
            "tokens_evaluated": total_tokens,
        }


__all__ = [
    "MetricAccumulator",
    "update_reconstruction_accumulator",
    "finalize_reconstruction_metrics",
    "compute_diversity_metrics",
    "normalize_answer",
    "normalized_exact_match",
    "normalized_token_f1",
    "compute_prompt_qa_metrics",
    "ExternalPPLEvaluator",
]
