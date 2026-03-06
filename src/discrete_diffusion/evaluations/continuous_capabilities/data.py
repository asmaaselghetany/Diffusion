"""Data loading utilities for continuous capability evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class FixedTextRecord:
    """One fixed-text record for reconstruction/infill evaluation."""

    id: str
    text: str


@dataclass
class QAPromptRecord:
    """One prompt-answering evaluation record."""

    id: str
    prompt: str
    answer: str
    max_answer_tokens: Optional[int] = None
    stop_strings: Optional[List[str]] = None
    suffix: Optional[str] = None



def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    rows: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx} in {file_path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_idx} in {file_path}")
            rows.append(row)
    return rows



def load_fixed_text_records(path: str | Path, max_records: Optional[int] = None) -> List[FixedTextRecord]:
    """Load fixed text records from JSONL.

    Expected schema per line:
        {"id": "<string>", "text": "<string>"}
    """
    rows = _read_jsonl(path)
    out: List[FixedTextRecord] = []

    for idx, row in enumerate(rows):
        if "text" not in row:
            raise ValueError(f"Fixed-text row {idx} missing required field 'text'")
        rec_id = str(row.get("id", f"fixed_{idx}"))
        text = str(row["text"])
        out.append(FixedTextRecord(id=rec_id, text=text))

    if max_records is not None and max_records > 0:
        out = out[:max_records]
    return out



def _validate_stop_strings(value: Any, row_idx: int) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"QA row {row_idx} field 'stop_strings' must be a list of strings when provided"
        )
    return [str(item) for item in value]



def load_qa_prompt_records(
    path: str | Path,
    max_records: Optional[int] = None,
) -> List[QAPromptRecord]:
    """Load prompt-answering records from JSONL.

    Required fields:
      - id
      - prompt
      - answer

    Optional fields:
      - max_answer_tokens
      - stop_strings
      - suffix
    """
    rows = _read_jsonl(path)
    out: List[QAPromptRecord] = []

    for idx, row in enumerate(rows):
        missing = [key for key in ("id", "prompt", "answer") if key not in row]
        if missing:
            raise ValueError(f"QA row {idx} missing required fields: {missing}")

        max_answer_tokens = row.get("max_answer_tokens")
        if max_answer_tokens is not None:
            if not isinstance(max_answer_tokens, int) or max_answer_tokens <= 0:
                raise ValueError(
                    f"QA row {idx} field 'max_answer_tokens' must be a positive integer when provided"
                )

        out.append(
            QAPromptRecord(
                id=str(row["id"]),
                prompt=str(row["prompt"]),
                answer=str(row["answer"]),
                max_answer_tokens=max_answer_tokens,
                stop_strings=_validate_stop_strings(row.get("stop_strings"), idx),
                suffix=str(row["suffix"]) if row.get("suffix") is not None else None,
            )
        )

    if max_records is not None and max_records > 0:
        out = out[:max_records]
    return out



def tokenize_fixed_text_records(
    tokenizer,
    records: Sequence[FixedTextRecord],
    *,
    max_length: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Tokenize fixed text records into tensors and metadata."""
    if not records:
        return {
            "source": "fixed_texts",
            "sample_ids": [],
            "texts": [],
            "input_ids": torch.empty(0, max_length, dtype=torch.long, device=device),
            "attention_mask": torch.empty(0, max_length, dtype=torch.long, device=device),
        }

    texts = [rec.text for rec in records]
    sample_ids = [rec.id for rec in records]

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    return {
        "source": "fixed_texts",
        "sample_ids": sample_ids,
        "texts": texts,
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }



def _slice_batch(batch: Dict[str, torch.Tensor], batch_take: int) -> Dict[str, torch.Tensor]:
    return {key: value[:batch_take] for key, value in batch.items()}



def load_validation_token_samples(
    config,
    tokenizer,
    *,
    max_samples: int,
    device: torch.device,
    seed: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load tokenized validation samples through the repo dataloader pipeline.

    Returns:
        (source_dict_or_none, error_message_or_none)
    """
    from ...data.loaders import get_dataloaders

    try:
        _, valid_loader = get_dataloaders(
            config,
            tokenizer,
            skip_train=True,
            skip_valid=False,
            valid_seed=seed,
        )
    except Exception as exc:
        return None, f"failed to build validation dataloader: {exc}"

    if valid_loader is None:
        return None, "validation dataloader is None"

    input_batches: List[torch.Tensor] = []
    attn_batches: List[torch.Tensor] = []
    remaining = max_samples

    try:
        for batch in valid_loader:
            if remaining <= 0:
                break
            take = min(remaining, int(batch["input_ids"].shape[0]))
            sliced = _slice_batch(batch, take)
            input_batches.append(sliced["input_ids"].to(device))
            attn_batches.append(sliced["attention_mask"].to(device))
            remaining -= take
    except Exception as exc:
        return None, f"failed while iterating validation dataloader: {exc}"

    if not input_batches:
        return None, "validation dataloader yielded no samples"

    input_ids = torch.cat(input_batches, dim=0)
    attention_mask = torch.cat(attn_batches, dim=0)

    sample_count = int(input_ids.shape[0])
    sample_ids = [f"valid_{idx}" for idx in range(sample_count)]

    return {
        "source": "validation",
        "sample_ids": sample_ids,
        "texts": None,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, None



def choose_source_subset(
    source: Dict[str, Any],
    *,
    num_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Return a deterministic subset of a source dictionary."""
    total = int(source["input_ids"].shape[0])
    if num_samples >= total:
        return source

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=num_samples, replace=False))
    idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=source["input_ids"].device)

    texts = source.get("texts")
    sample_ids = source.get("sample_ids")

    subset = {
        "source": source["source"],
        "input_ids": source["input_ids"].index_select(0, idx_tensor),
        "attention_mask": source["attention_mask"].index_select(0, idx_tensor),
        "sample_ids": [sample_ids[i] for i in indices] if sample_ids is not None else None,
        "texts": [texts[i] for i in indices] if texts is not None else None,
    }
    return subset


__all__ = [
    "FixedTextRecord",
    "QAPromptRecord",
    "load_fixed_text_records",
    "load_qa_prompt_records",
    "tokenize_fixed_text_records",
    "load_validation_token_samples",
    "choose_source_subset",
]
