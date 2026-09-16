"""Race-safe on-disk dataset cache helpers.

Nemotron tokenize / fingerprint / split logic lives in ``loaders.py`` and
``conversion_baseline`` (single source of truth). This module only owns
lock / sentinel / atomic save-load for ``.dat`` caches.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
import uuid

import datasets

from .. import utils

LOGGER = utils.get_logger(__name__)

_CACHE_SENTINEL = '.discrete_diffusion_complete.json'
_CACHE_SENTINEL_FORMAT = 'discrete-diffusion-dataset-cache'
_CACHE_SENTINEL_VERSION = 1
_CACHE_LOCK_OWNER = 'owner.json'
_CACHE_STALE_SECONDS_DEFAULT = 7 * 24 * 60 * 60
_CACHE_LOCK_POLL_SECONDS_DEFAULT = 10.0


def _cache_stale_seconds() -> float:
  raw = os.environ.get('DISCRETE_DIFFUSION_CACHE_STALE_SECONDS')
  return float(raw) if raw is not None else _CACHE_STALE_SECONDS_DEFAULT


def _cache_lock_poll_seconds() -> float:
  raw = os.environ.get('DISCRETE_DIFFUSION_CACHE_LOCK_POLL_SECONDS')
  return float(raw) if raw is not None else _CACHE_LOCK_POLL_SECONDS_DEFAULT


def _read_json(path: str) -> dict:
  with open(path, encoding='utf-8') as handle:
    value = json.load(handle)
  if not isinstance(value, dict):
    raise ValueError(f'Expected a JSON object in {path}')
  return value


def _fsync_directory(path: str) -> None:
  """Best-effort metadata flush before an atomic directory rename."""
  try:
    fd = os.open(path, os.O_RDONLY)
  except OSError:
    return
  try:
    os.fsync(fd)
  except OSError:
    # Some network filesystems do not support fsync on directories.
    pass
  finally:
    os.close(fd)


def _write_json_atomic(path: str, value: dict) -> None:
  temporary = f'{path}.tmp.{uuid.uuid4().hex}'
  try:
    with open(temporary, 'x', encoding='utf-8') as handle:
      json.dump(value, handle, sort_keys=True)
      handle.write('\n')
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(os.path.dirname(path))
  finally:
    try:
      os.unlink(temporary)
    except FileNotFoundError:
      pass


def _legacy_dataset_cache_manifest_is_complete(path: str) -> bool:
  """Check the files that ``Dataset.save_to_disk`` commits at completion.

  Older caches do not have our sentinel. They remain usable only if the HF
  state and info manifests exist and every data shard named by state exists.
  Loading the dataset is an additional validation step below.
  """
  state_path = os.path.join(path, 'state.json')
  info_path = os.path.join(path, 'dataset_info.json')
  if not (os.path.isfile(state_path) and os.path.isfile(info_path)):
    return False
  try:
    state = _read_json(state_path)
    data_files = state['_data_files']
    if not isinstance(data_files, list):
      return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get('filename'), str)
        and os.path.isfile(os.path.join(path, item['filename']))
        for item in data_files)
  except (KeyError, OSError, ValueError, json.JSONDecodeError):
    return False


def _load_complete_dataset_cache(path: str):
  """Return a validated cache, or None for an absent/incomplete legacy path."""
  if not os.path.lexists(path):
    return None
  if not os.path.isdir(path):
    raise RuntimeError(f'Dataset cache path is not a directory: {path}')

  sentinel_path = os.path.join(path, _CACHE_SENTINEL)
  has_sentinel = os.path.isfile(sentinel_path)
  sentinel = None
  if has_sentinel:
    try:
      sentinel = _read_json(sentinel_path)
      if sentinel.get('format') != _CACHE_SENTINEL_FORMAT:
        raise ValueError('unrecognized cache format')
      if sentinel.get('version') != _CACHE_SENTINEL_VERSION:
        raise ValueError('unsupported cache sentinel version')
    except (OSError, ValueError, json.JSONDecodeError) as error:
      raise RuntimeError(
          f'Dataset cache has an invalid completion sentinel: {path}') from error
  elif not _legacy_dataset_cache_manifest_is_complete(path):
    return None

  try:
    dataset = datasets.load_from_disk(path)
    length = len(dataset)
    columns = list(dataset.column_names)
  except Exception as error:
    if has_sentinel:
      raise RuntimeError(
          f'Dataset cache is marked complete but failed validation: {path}') from error
    LOGGER.warning('Ignoring incomplete legacy dataset cache %s: %s', path, error)
    return None

  if sentinel is not None:
    if sentinel.get('length') != length or sentinel.get('columns') != columns:
      raise RuntimeError(
          f'Dataset cache completion metadata does not match contents: {path}')
  else:
    LOGGER.info('Using validated legacy dataset cache without sentinel: %s', path)
  return dataset


def _latest_tree_mtime(path: str) -> float:
  latest = os.lstat(path).st_mtime
  if not os.path.isdir(path) or os.path.islink(path):
    return latest
  for root, dirs, files in os.walk(path):
    for name in dirs + files:
      try:
        latest = max(latest, os.lstat(os.path.join(root, name)).st_mtime)
      except FileNotFoundError:
        # A legacy writer may still be updating the directory.
        return time.time()
  return latest


def _recovery_path(path: str, reason: str) -> str:
  stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
  return f'{path}.recovered-{reason}.{stamp}.{uuid.uuid4().hex}'


def _recover_stale_incomplete_cache(path: str) -> None:
  """Move an old partial target aside, never deleting possibly useful data."""
  age = time.time() - _latest_tree_mtime(path)
  stale_seconds = _cache_stale_seconds()
  if age <= stale_seconds:
    raise RuntimeError(
        f'Dataset cache exists but is not complete: {path}. It was modified '
        f'{age:.0f}s ago; refusing to alter a possibly active legacy writer. '
        'Wait for that writer or move the partial directory aside manually.')
  recovery = _recovery_path(path, 'incomplete-cache')
  os.rename(path, recovery)
  LOGGER.warning('Moved stale incomplete cache %s to recoverable path %s',
                 path, recovery)


def _recover_legacy_flock_file(lock_path: str, cache_path: str) -> None:
  """Conservatively migrate the regular file used by the former protocol.

  The old implementation never removed its flock file, and flock ownership is
  not reliably observable across all NFS clients. Therefore absence of a local
  flock is not considered proof of safety: both this file and any incomplete
  target must have exceeded the stale window before the file is moved aside.
  """
  if not os.path.lexists(lock_path):
    return
  if not os.path.isfile(lock_path) or os.path.islink(lock_path):
    raise RuntimeError(
        f'Legacy dataset cache lock is not a regular file: {lock_path}')
  stale_seconds = _cache_stale_seconds()
  lock_age = time.time() - os.lstat(lock_path).st_mtime
  target_age = None
  if os.path.lexists(cache_path):
    target_age = time.time() - _latest_tree_mtime(cache_path)
  if lock_age <= stale_seconds or (
      target_age is not None and target_age <= stale_seconds):
    detail = f'legacy lock age={lock_age:.0f}s'
    if target_age is not None:
      detail += f', target age={target_age:.0f}s'
    raise RuntimeError(
        f'Legacy dataset cache lock may still have an active writer: '
        f'{lock_path} ({detail}). Refusing to alter it; wait for the old '
        'writer or move it aside after confirming that writer has exited.')
  recovery = _recovery_path(lock_path, 'legacy-flock')
  try:
    os.rename(lock_path, recovery)
  except FileNotFoundError:
    # Another new-protocol process may have completed the same migration.
    return
  LOGGER.warning('Moved stale legacy flock file %s to recoverable path %s',
                 lock_path, recovery)


def _lock_owner(lock_path: str) -> dict | None:
  try:
    return _read_json(os.path.join(lock_path, _CACHE_LOCK_OWNER))
  except (OSError, ValueError, json.JSONDecodeError):
    return None


def _pid_is_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def _lock_is_stale(lock_path: str) -> bool:
  try:
    age = time.time() - _latest_tree_mtime(lock_path)
  except FileNotFoundError:
    return False
  if age <= _cache_stale_seconds():
    return False
  owner = _lock_owner(lock_path)
  if owner and owner.get('hostname') == socket.gethostname():
    try:
      pid = int(owner['pid'])
    except (KeyError, TypeError, ValueError):
      pass
    else:
      if _pid_is_alive(pid):
        return False
  return True


def _owns_cache_lock(lock_path: str, token: str) -> bool:
  owner = _lock_owner(lock_path)
  return owner is not None and owner.get('token') == token


def _lock_observation(lock_path: str):
  """Identity used while conservatively confirming a stale lock."""
  stat = os.stat(lock_path)
  owner = _lock_owner(lock_path)
  return (
      stat.st_dev,
      stat.st_ino,
      _latest_tree_mtime(lock_path),
      owner.get('token') if owner else None,
  )


def _reclaim_stale_cache_lock(lock_path: str, expected_observation,
                              poll_seconds: float) -> bool:
  """Reclaim the observed lock while excluding competing reclaimers.

  Every conforming acquirer waits while the reaper guard exists. Thus, after
  acquiring the guard and confirming the same inode/token twice, the pathname
  cannot be replaced by a fresh conforming lock immediately before rename.
  """
  reaper_path = f'{lock_path}.reaping'
  try:
    os.mkdir(reaper_path)
  except FileExistsError:
    # Reaping takes only one poll interval. Do not guess that an abandoned
    # reaper guard is safe to break; it can be moved aside manually.
    return False
  try:
    try:
      current = _lock_observation(lock_path)
    except FileNotFoundError:
      return False
    if current != expected_observation or not _lock_is_stale(lock_path):
      return False
    time.sleep(poll_seconds)
    try:
      confirmed = _lock_observation(lock_path)
    except FileNotFoundError:
      return False
    if confirmed != current or not _lock_is_stale(lock_path):
      return False
    recovery = _recovery_path(lock_path, 'stale-lock')
    os.rename(lock_path, recovery)
    LOGGER.warning('Reclaimed stale cache lock %s as %s', lock_path, recovery)
    return True
  finally:
    try:
      os.rmdir(reaper_path)
    except FileNotFoundError:
      pass


@contextlib.contextmanager
def _exclusive_dataset_cache_lock(lock_path: str):
  """Serialize cache generation with atomic ``mkdir`` on shared filesystems.

  A timed-out lock is atomically moved aside rather than deleted. The owner
  token also fences a reclaimed writer from publishing its staging directory.
  """
  parent = os.path.dirname(lock_path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  token = uuid.uuid4().hex
  stale_observation = None
  poll_seconds = _cache_lock_poll_seconds()
  while True:
    # A stale-lock reclaimer gets a tiny critical section spanning its final
    # identity check and rename. This closes the replace-before-rename race.
    reaper_path = f'{lock_path}.reaping'
    if os.path.lexists(reaper_path):
      try:
        reaper_age = time.time() - _latest_tree_mtime(reaper_path)
      except FileNotFoundError:
        continue
      if reaper_age > _cache_stale_seconds():
        raise RuntimeError(
            f'Stale dataset-cache reaper guard found at {reaper_path}. '
            'Refusing an automatic takeover; move it aside after confirming '
            'that no cache-lock recovery is active.')
      time.sleep(poll_seconds)
      continue
    try:
      os.mkdir(lock_path)
      break
    except FileExistsError:
      if not os.path.isdir(lock_path):
        raise RuntimeError(
            f'Dataset cache lock path is not a directory: {lock_path}')
      if _lock_is_stale(lock_path):
        try:
          observation = _lock_observation(lock_path)
        except FileNotFoundError:
          continue
        # Require two unchanged observations. This avoids reclaiming a lock
        # whose heartbeat became visible just as its lease appeared stale.
        if stale_observation == observation:
          _reclaim_stale_cache_lock(lock_path, observation, poll_seconds)
          stale_observation = None
          continue
        stale_observation = observation
      else:
        stale_observation = None
      time.sleep(poll_seconds)

  owner = {
      'token': token,
      'hostname': socket.gethostname(),
      'pid': os.getpid(),
      'created_at': time.time(),
  }
  try:
    _write_json_atomic(os.path.join(lock_path, _CACHE_LOCK_OWNER), owner)
  except Exception:
    # We created this directory and have not yielded ownership yet.
    try:
      os.rmdir(lock_path)
    except OSError:
      pass
    raise
  stop_heartbeat = threading.Event()

  def _heartbeat():
    interval = max(1.0, min(60.0, _cache_stale_seconds() / 3.0))
    while not stop_heartbeat.wait(interval):
      if not _owns_cache_lock(lock_path, token):
        return
      try:
        os.utime(lock_path, None)
      except FileNotFoundError:
        return

  heartbeat = threading.Thread(
      target=_heartbeat, name='dataset-cache-lock-heartbeat', daemon=True)
  heartbeat.start()
  try:
    yield token
  finally:
    stop_heartbeat.set()
    heartbeat.join(timeout=1.0)
    if _owns_cache_lock(lock_path, token):
      try:
        os.unlink(os.path.join(lock_path, _CACHE_LOCK_OWNER))
        os.rmdir(lock_path)
      except FileNotFoundError:
        pass


def _save_dataset_cache_atomically(dataset, path: str, lock_path: str,
                                   lock_token: str):
  """Validate in a sibling directory, mark complete, then atomically publish."""
  staging = (
      f'{path}.building.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}')
  dataset.save_to_disk(staging)
  staged = _load_complete_dataset_cache(staging)
  if staged is None:
    raise RuntimeError(f'New dataset cache failed pre-publish validation: {staging}')
  sentinel = {
      'format': _CACHE_SENTINEL_FORMAT,
      'version': _CACHE_SENTINEL_VERSION,
      'length': len(staged),
      'columns': list(staged.column_names),
      'created_at': time.time(),
  }
  _write_json_atomic(os.path.join(staging, _CACHE_SENTINEL), sentinel)
  _load_complete_dataset_cache(staging)
  _fsync_directory(staging)
  if not _owns_cache_lock(lock_path, lock_token):
    raise RuntimeError(
        f'Lost ownership of dataset cache lock before publish: {lock_path}; '
        f'validated staging data remains at {staging}')
  if os.path.lexists(path):
    winner = _load_complete_dataset_cache(path)
    if winner is not None:
      LOGGER.info('Another writer published dataset cache first: %s', path)
      return winner
    raise RuntimeError(
        f'Dataset cache target appeared incomplete during generation: {path}; '
        f'validated staging data remains at {staging}')
  os.rename(staging, path)
  _fsync_directory(os.path.dirname(path))
  return dataset


