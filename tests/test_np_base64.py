"""np_to_base64 / base64_to_np use numpy .npy (no pickle by default)."""

from __future__ import annotations

import base64
import pickle

import numpy as np
import pytest

from discrete_diffusion.utils.utils import base64_to_np, np_to_base64


def test_np_base64_roundtrip():
  arr = np.arange(12, dtype=np.int64).reshape(3, 4)
  out = base64_to_np(np_to_base64(arr))
  assert out.dtype == arr.dtype
  np.testing.assert_array_equal(out, arr)


def test_np_base64_is_npy_magic():
  raw = base64.b64decode(np_to_base64(np.zeros(2)))
  assert raw[:6] == b'\x93NUMPY'


def test_legacy_pickle_requires_env_gate(monkeypatch):
  payload = base64.b64encode(pickle.dumps(np.ones(3))).decode('ascii')
  monkeypatch.delenv('ALLOW_PICKLE_SAMPLES', raising=False)
  with pytest.raises(ValueError, match='ALLOW_PICKLE_SAMPLES'):
    base64_to_np(payload)
  monkeypatch.setenv('ALLOW_PICKLE_SAMPLES', '1')
  np.testing.assert_array_equal(base64_to_np(payload), np.ones(3))
