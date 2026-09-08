"""Regression tests for cache creation and block-aligned SFT packing."""

import json
import os
import socket
import threading
import time
from unittest import mock

import datasets
import pytest
import torch

from discrete_diffusion.data import dataset_cache as dc
from discrete_diffusion.data import loaders
from discrete_diffusion.data.processing import _group_block_aligned_sft
from discrete_diffusion.data.tokenizers import Text8Tokenizer


def test_get_dataset_builds_then_reuses_fresh_cache(tmp_path):
  raw = datasets.DatasetDict({
      'train': datasets.Dataset.from_dict({
          'text': [
              'abcdefghijklmnopqrstuvwxyz',
              'the quick brown fox jumps over the lazy dog',
          ],
      }),
  })

  with mock.patch.object(
      loaders.datasets, 'load_dataset', return_value=raw) as load_dataset:
    first = loaders.get_dataset(
        'unit-fixture', Text8Tokenizer(), wrap=True, mode='train',
        cache_dir=str(tmp_path), block_size=8, num_proc=1,
        streaming=False)
    second = loaders.get_dataset(
        'unit-fixture', Text8Tokenizer(), wrap=True, mode='train',
        cache_dir=str(tmp_path), block_size=8, num_proc=1,
        streaming=False)

  assert len(first) > 0
  assert first[0]['input_ids'].shape == (8,)
  assert torch.equal(first[0]['input_ids'], second[0]['input_ids'])
  assert load_dataset.call_count == 1
  cache = tmp_path / 'unit-fixture_train_bs8_wrapped.dat'
  assert cache.is_dir()
  assert (cache / dc._CACHE_SENTINEL).is_file()
  assert not list(tmp_path.glob('*.building.*'))
  assert not list(tmp_path.glob('*.lockdir'))


def test_atomic_cache_publish_never_exposes_staging_directory(tmp_path):
  dataset = datasets.Dataset.from_dict({'input_ids': [[1, 2, 3]]})
  target = tmp_path / 'processed.dat'
  lock = tmp_path / 'processed.dat.lockdir'
  original_save = dataset.save_to_disk

  def observed_save(path, *args, **kwargs):
    assert str(path).startswith(f'{target}.building.')
    assert not target.exists()
    original_save(path, *args, **kwargs)
    assert not target.exists()

  with mock.patch.object(dataset, 'save_to_disk', side_effect=observed_save):
    with dc._exclusive_dataset_cache_lock(str(lock)) as token:
      dc._save_dataset_cache_atomically(
          dataset, str(target), str(lock), token)

  assert target.is_dir()
  assert (target / dc._CACHE_SENTINEL).is_file()
  assert len(dc._load_complete_dataset_cache(str(target))) == 1


def test_mkdir_cache_lock_excludes_concurrent_threads(tmp_path, monkeypatch):
  monkeypatch.setenv('DISCRETE_DIFFUSION_CACHE_LOCK_POLL_SECONDS', '0.01')
  lock = str(tmp_path / 'cache.lockdir')
  first_entered = threading.Event()
  release_first = threading.Event()
  second_entered = threading.Event()

  def first():
    with dc._exclusive_dataset_cache_lock(lock):
      first_entered.set()
      assert release_first.wait(timeout=5)

  def second():
    assert first_entered.wait(timeout=5)
    with dc._exclusive_dataset_cache_lock(lock):
      second_entered.set()

  first_thread = threading.Thread(target=first)
  second_thread = threading.Thread(target=second)
  first_thread.start()
  second_thread.start()
  assert first_entered.wait(timeout=5)
  assert not second_entered.wait(timeout=0.1)
  release_first.set()
  first_thread.join(timeout=5)
  second_thread.join(timeout=5)
  assert not first_thread.is_alive()
  assert not second_thread.is_alive()
  assert second_entered.is_set()


def test_stale_lock_is_moved_aside_not_deleted(tmp_path, monkeypatch):
  monkeypatch.setenv('DISCRETE_DIFFUSION_CACHE_STALE_SECONDS', '0.05')
  monkeypatch.setenv('DISCRETE_DIFFUSION_CACHE_LOCK_POLL_SECONDS', '0.01')
  lock = tmp_path / 'cache.lockdir'
  lock.mkdir()
  owner = lock / dc._CACHE_LOCK_OWNER
  owner.write_text(json.dumps({
      'token': 'abandoned',
      'hostname': 'some-other-node',
      'pid': 12345,
      'created_at': 1,
  }))
  old = time.time() - 10
  os.utime(owner, (old, old))
  os.utime(lock, (old, old))

  with dc._exclusive_dataset_cache_lock(str(lock)) as token:
    assert dc._owns_cache_lock(str(lock), token)

  recovered = list(tmp_path.glob('cache.lockdir.recovered-stale-lock.*'))
  assert len(recovered) == 1
  assert json.loads(
      (recovered[0] / dc._CACHE_LOCK_OWNER).read_text())['token'] == 'abandoned'
