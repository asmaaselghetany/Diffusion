"""Training utilities (init checks, verification helpers)."""

from .init import (
    compute_ar_block_init_metrics,
    log_ar_block_init_metrics,
    logit_agreement_rho,
    top1_agreement,
)

__all__ = [
    'compute_ar_block_init_metrics',
    'log_ar_block_init_metrics',
    'logit_agreement_rho',
    'top1_agreement',
]
