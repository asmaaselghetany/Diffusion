"""PYTHONSTARTUP hook: patch Triton before Inductor worker imports."""
from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
_src = _repo / "src"
if _src.is_dir() and str(_src) not in sys.path:
  sys.path.insert(0, str(_src))

try:
  from discrete_diffusion.compat.triton_shim import ensure_triton_attrs_descriptor
  ensure_triton_attrs_descriptor()
except Exception:
  pass
