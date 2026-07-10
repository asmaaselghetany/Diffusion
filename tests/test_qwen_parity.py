"""Optional Qwen parity tests (network + GPU)."""

import os

import pytest

pytestmark = pytest.mark.qwen


@pytest.mark.skipif(
    os.environ.get('RUN_QWEN_TESTS', '') != '1',
    reason='set RUN_QWEN_TESTS=1 to download Qwen',
)
def test_qwen_load_fraction():
  from scripts.verify_qwen_load import main
  assert main() == 0


@pytest.mark.skipif(
    os.environ.get('RUN_QWEN_TESTS', '') != '1',
    reason='set RUN_QWEN_TESTS=1 to download Qwen',
)
def test_block_forward_differs_from_causal():
  from scripts.verify_block_forward import main
  assert main() == 0
