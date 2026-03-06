"""Embedding provider abstractions for continuous text diffusion."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Map token ids <-> embedding/logit spaces."""

    vocab_size: int
    dim: int

    def embed(self, input_ids: LongTensor, attention_mask: Optional[LongTensor] = None) -> FloatTensor:
        ...

    def logits(self, embeddings: FloatTensor) -> FloatTensor:
        ...


class LookupEmbeddingProvider(nn.Module):
    """Simple embedding table with optional untied readout."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        trainable: bool = True,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)

        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.tie_weights = bool(tie_weights)
        if not self.tie_weights:
            self.readout = nn.Linear(self.dim, self.vocab_size)
            nn.init.normal_(self.readout.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.readout.bias)
        else:
            self.readout = None
            self.bias = nn.Parameter(torch.zeros(self.vocab_size))

        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    def embed(self, input_ids: LongTensor, attention_mask: Optional[LongTensor] = None) -> FloatTensor:
        del attention_mask
        return self.embedding(input_ids)

    def logits(self, embeddings: FloatTensor) -> FloatTensor:
        if self.tie_weights:
            return F.linear(embeddings, self.embedding.weight, self.bias)
        assert self.readout is not None
        return self.readout(embeddings)


class TiedEmbeddingProvider(LookupEmbeddingProvider):
    """Embedding provider with tied readout weights."""

    def __init__(self, vocab_size: int, dim: int, *, trainable: bool = True) -> None:
        super().__init__(vocab_size=vocab_size, dim=dim, trainable=trainable, tie_weights=True)


class ContextualEmbeddingProviderAdapter(nn.Module):
    """Adapter wrapping contextual encoders for backward compatibility."""

    def __init__(self, encoder: nn.Module, vocab_size: int, dim: int, decoder: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.encoder = encoder
        self._decoder = decoder
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)

    def embed(self, input_ids: LongTensor, attention_mask: Optional[LongTensor] = None) -> FloatTensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.encoder(input_ids, attention_mask)

    def logits(self, embeddings: FloatTensor) -> FloatTensor:
        if self._decoder is None:
            raise RuntimeError(
                "ContextualEmbeddingProviderAdapter requires a decoder for logits(). "
                "Set legacy decoder or use a lookup/tied provider."
            )
        return self._decoder(embeddings)


def build_lookup_or_tied_provider(
    *,
    provider_name: str,
    vocab_size: int,
    dim: int,
    trainable: bool,
) -> EmbeddingProvider:
    provider_name = str(provider_name).lower()
    if provider_name == "lookup":
        return LookupEmbeddingProvider(vocab_size=vocab_size, dim=dim, trainable=trainable, tie_weights=False)
    if provider_name == "tied":
        return TiedEmbeddingProvider(vocab_size=vocab_size, dim=dim, trainable=trainable)
    raise ValueError(f"Unknown provider_name: {provider_name}")


__all__ = [
    "EmbeddingProvider",
    "LookupEmbeddingProvider",
    "TiedEmbeddingProvider",
    "ContextualEmbeddingProviderAdapter",
    "build_lookup_or_tied_provider",
]
