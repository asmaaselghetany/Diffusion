"""Sampling helpers for discrete diffusion algorithms."""

from __future__ import annotations

from .absorbing import AbsorbingSampler
from .ar import ARSampler
from .base import Sampler
from .bd3lm import BD3LMSampler
from .continuous_embedding import ContinuousEmbeddingSampler
from .gidd import GIDDSampler
from .latent_jepa import LatentJEPASampler
from .partition import PartitionSampler
from .position_scorer import (
  ConfidencePositionScorer,
  EntropyPositionScorer,
  MarginPositionScorer,
  PositionScoringCriteria,
  RandomPositionScorer,
)
from .token_selection import (
  GreedySelection,
  NucleusSelection,
  TemperatureSelection,
  TokenSelectionCriteria,
  TopKSelection,
)
from .uniform import UniformSampler

__all__ = [
  "Sampler",
  "AbsorbingSampler",
  "ARSampler",
  "BD3LMSampler",
  "ContinuousEmbeddingSampler",
  "GIDDSampler",
  "LatentJEPASampler",
  "PartitionSampler",
  "UniformSampler",
  # Position scorers
  "PositionScoringCriteria",
  "RandomPositionScorer",
  "ConfidencePositionScorer",
  "EntropyPositionScorer",
  "MarginPositionScorer",
  # Token selection
  "TokenSelectionCriteria",
  "GreedySelection",
  "TemperatureSelection",
  "TopKSelection",
  "NucleusSelection",
]
