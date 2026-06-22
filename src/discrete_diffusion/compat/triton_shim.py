"""Triton API shims for PyTorch flex_attention on aarch64.

PyTorch 2.6 cu126 wheels ship a Triton build without ``AttrsDescriptor``,
but flex_attention / torch.compile still import it at runtime.  Patch the
symbol in before Inductor loads.
"""

from __future__ import annotations

import collections
import sys
import types
from typing import Any, Dict, Tuple


def _attrs_descriptor_class():
  class AttrsDescriptor(
      collections.namedtuple(
          "AttrsDescriptor",
          ["divisible_by_16", "equal_to_1"],
          defaults=((), ()),
      )
  ):
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AttrsDescriptor":
      props = payload.get("arg_properties", payload)
      return cls(
          divisible_by_16=tuple(props.get("tt.divisibility", props.get("divisible_by_16", ()))),
          equal_to_1=tuple(props.get("tt.equal_to", props.get("equal_to_1", ()))),
      )

    @property
    def property_values(self) -> Dict[str, int]:
      return {"tt.divisibility": 16, "tt.equal_to": 1}

  return AttrsDescriptor


def ensure_triton_attrs_descriptor() -> bool:
  """Return True if AttrsDescriptor is available (native or patched)."""
  try:
    import triton  # noqa: F401
  except ImportError:
    return False

  import triton.backends.compiler as backends_compiler
  import triton.compiler.compiler as compiler

  if hasattr(compiler, "AttrsDescriptor"):
    return True

  attrs = _attrs_descriptor_class()
  compiler.AttrsDescriptor = attrs
  backends_compiler.AttrsDescriptor = attrs
  return True


def install() -> None:
  if not ensure_triton_attrs_descriptor():
    return
  # Re-import hints so Inductor picks up patched AttrsDescriptor.
  for name in list(sys.modules):
    if name == "torch._inductor.runtime.hints" or name.startswith("torch._inductor.runtime.hints."):
      del sys.modules[name]
