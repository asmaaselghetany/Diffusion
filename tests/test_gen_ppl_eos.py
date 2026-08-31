"""Generative PPL EOS trimming."""

import numpy as np

from discrete_diffusion.evaluations.generative_ppl import _trim_token_rows_at_eos


def test_trim_token_rows_keeps_eos_for_first_chunk():
  eos = 99
  rows = np.array([[1, 2, eos, 3, 4], [5, 6, 7, eos, 8]], dtype=np.int64)
  out = _trim_token_rows_at_eos(rows, eos)
  assert out.shape[0] == 2
  assert out[0, 2] == eos
  assert out[0, 3] == 0
  assert out[1, 3] == eos
