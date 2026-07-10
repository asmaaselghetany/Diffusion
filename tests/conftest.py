"""Pytest configuration for local smoke tests."""

import pytest


def pytest_configure(config):
  config.addinivalue_line(
      'markers',
      'qwen: tests that download/load a Qwen checkpoint (set RUN_QWEN_TESTS=1)',
  )


def pytest_collection_modifyitems(config, items):
  import os
  if os.environ.get('RUN_QWEN_TESTS', '0') == '1':
    return
  skip = pytest.mark.skip(
      reason='Qwen tests skipped (set RUN_QWEN_TESTS=1)')
  for item in items:
    if 'qwen' in item.keywords:
      item.add_marker(skip)
