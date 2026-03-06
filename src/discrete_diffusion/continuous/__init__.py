"""Continuous embedding diffusion/flow primitives."""

from .math import (
    VPSchedule,
    build_vp_schedule,
    build_vp_schedule_from_alphabar,
    scale_alphabar_noise,
    q_sample,
    x0_from_eps,
    eps_from_x0,
    v_from_x0_eps,
    x0_from_v,
    eps_from_v,
    q_posterior,
)
from .embeddings import (
    EmbeddingProvider,
    LookupEmbeddingProvider,
    TiedEmbeddingProvider,
    ContextualEmbeddingProviderAdapter,
    build_lookup_or_tied_provider,
)
from .interpolants import Interpolant, VPInterpolant, RectifiedInterpolant
from .objectives import (
    Objective,
    ObjectiveOutput,
    DDPMObjective,
    DDPMObjectiveConfig,
    FlowMapObjective,
    FlowMapObjectiveConfig,
    FlowMatchingObjective,
    FlowMatchingObjectiveConfig,
    IMFObjective,
    IMFObjectiveConfig,
)
from .sampling import VPSampler, RectifiedSampler
from .conditioning import SpanMasker, SpanMaskerConfig
from .diagnostics import run_sampler_smoke, probe_model_jvp

__all__ = [
    "VPSchedule",
    "build_vp_schedule",
    "build_vp_schedule_from_alphabar",
    "scale_alphabar_noise",
    "q_sample",
    "x0_from_eps",
    "eps_from_x0",
    "v_from_x0_eps",
    "x0_from_v",
    "eps_from_v",
    "q_posterior",
    "EmbeddingProvider",
    "LookupEmbeddingProvider",
    "TiedEmbeddingProvider",
    "ContextualEmbeddingProviderAdapter",
    "build_lookup_or_tied_provider",
    "Interpolant",
    "VPInterpolant",
    "RectifiedInterpolant",
    "Objective",
    "ObjectiveOutput",
    "DDPMObjective",
    "DDPMObjectiveConfig",
    "FlowMapObjective",
    "FlowMapObjectiveConfig",
    "FlowMatchingObjective",
    "FlowMatchingObjectiveConfig",
    "IMFObjective",
    "IMFObjectiveConfig",
    "VPSampler",
    "RectifiedSampler",
    "SpanMasker",
    "SpanMaskerConfig",
    "run_sampler_smoke",
    "probe_model_jvp",
]
