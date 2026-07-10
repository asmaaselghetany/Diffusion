from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ForwardMode = Literal['causal', 'block_diff']


@dataclass
class QwenBlockConfig:
  hub_id: str = 'Qwen/Qwen2.5-0.5B'
  block_size: int = 16
  length: int = 128
  forward_mode: ForwardMode = 'block_diff'
  attn_implementation: str = 'eager'
