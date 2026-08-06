"""Skeleton contracts: special tokens, hook compatibility, lever policy."""

from .special_tokens import SpecialTokenIds, ensure_special_tokens, assert_same_mask_id
from .attention_hook import (
    assert_block_attention_hook_compatible,
    HOOK_MISSING_MSG,
)

__all__ = [
    'SpecialTokenIds',
    'ensure_special_tokens',
    'assert_same_mask_id',
    'assert_block_attention_hook_compatible',
    'HOOK_MISSING_MSG',
]
