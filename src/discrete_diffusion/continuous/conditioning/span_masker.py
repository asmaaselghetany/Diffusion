"""Span masking for infilling training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class SpanMaskerConfig:
    min_spans: int = 1
    max_spans: int = 3
    length_dist: str = "geometric"  # geometric | poisson
    mean_span_length: float = 4.0


class SpanMasker:
    """Sample span masks where True means "generate"."""

    def __init__(self, config: Optional[SpanMaskerConfig] = None) -> None:
        self.config = config or SpanMaskerConfig()
        if self.config.min_spans <= 0:
            raise ValueError("min_spans must be > 0")
        if self.config.max_spans < self.config.min_spans:
            raise ValueError("max_spans must be >= min_spans")

    def _sample_span_length(self, device: torch.device) -> int:
        if self.config.length_dist == "poisson":
            v = torch.poisson(torch.tensor(self.config.mean_span_length, device=device)).item()
            return max(1, int(v))
        if self.config.length_dist == "geometric":
            p = 1.0 / max(self.config.mean_span_length, 1.0)
            # Geometric returns #trials until first success.
            geo = torch.distributions.Geometric(probs=torch.tensor(p, device=device))
            return max(1, int(geo.sample().item()))
        raise ValueError(f"Unknown length_dist: {self.config.length_dist}")

    def sample_mask(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return span mask [B, L] with 1=generate and 0=condition."""
        del input_ids
        bsz, seqlen = attention_mask.shape
        span_mask = torch.zeros((bsz, seqlen), dtype=torch.bool, device=attention_mask.device)

        for b in range(bsz):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len <= 1:
                # Degenerate; keep one token conditioned if possible.
                if valid_len == 1:
                    span_mask[b, 0] = True
                continue

            num_spans = int(torch.randint(
                low=self.config.min_spans,
                high=self.config.max_spans + 1,
                size=(1,),
                device=attention_mask.device,
            ).item())

            for _ in range(num_spans):
                span_len = min(self._sample_span_length(attention_mask.device), max(valid_len - 1, 1))
                max_start = max(valid_len - span_len, 0)
                start = int(torch.randint(0, max_start + 1, (1,), device=attention_mask.device).item()) if max_start > 0 else 0
                end = min(valid_len, start + span_len)
                span_mask[b, start:end] = True

            # Guardrail: at least one masked and one unmasked valid token.
            valid_positions = attention_mask[b].bool()
            masked_valid = span_mask[b] & valid_positions
            if int(masked_valid.sum().item()) == 0:
                span_mask[b, 0] = True
                masked_valid = span_mask[b] & valid_positions
            if int(masked_valid.sum().item()) >= valid_len:
                # Unmask one valid token.
                span_mask[b, valid_len - 1] = False

        return span_mask

    def apply_mask_token(
        self,
        input_ids: Tensor,
        span_mask: Tensor,
        mask_token_id: int,
    ) -> Tensor:
        masked = input_ids.clone()
        masked[span_mask] = int(mask_token_id)
        return masked

    def make_batch(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        mask_token_id: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        span_mask = self.sample_mask(input_ids=input_ids, attention_mask=attention_mask)
        masked_input_ids = None
        if mask_token_id is not None:
            masked_input_ids = self.apply_mask_token(input_ids, span_mask, mask_token_id)
        return span_mask, masked_input_ids


__all__ = ["SpanMasker", "SpanMaskerConfig"]
