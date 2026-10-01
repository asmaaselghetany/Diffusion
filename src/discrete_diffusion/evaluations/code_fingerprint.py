"""Code fingerprints for eval result provenance.

Logs a SHA256 of critical forward-process sources so a truncated/stubbed
``utils.py`` (or silent checkout drift) shows up in the result JSON.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


# Relative to discrete_diffusion package root (…/src/discrete_diffusion).
_CRITICAL = (
    'forward_process/utils.py',
)


def _pkg_root() -> Path:
  return Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open('rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
      h.update(chunk)
  return h.hexdigest()


def forward_process_utils_fingerprint() -> dict:
  """Hash + basic shape checks for ``forward_process/utils.py``."""
  path = _pkg_root() / 'forward_process' / 'utils.py'
  sha = file_sha256(path) if path.is_file() else None
  n_lines = None
  if path.is_file():
    n_lines = sum(1 for _ in path.open('rb'))
  # Required symbols for conversion Unif (train + decode).
  required = (
      'normalize_uniform_simplex_mode',
      'resolve_uniform_exclude_ids',
      'resolve_uniform_noise_redraw_exclude_ids',
      'uniform_simplex_size',
      'sample_uniform_excluding_mask',
      'unused_embed_slot_ids',
  )
  missing = []
  try:
    from discrete_diffusion.forward_process import utils as u
    for name in required:
      if not hasattr(u, name):
        missing.append(name)
  except Exception as e:  # noqa: BLE001
    missing.append(f'import_error:{type(e).__name__}')
  return {
      'path': str(path),
      'sha256': sha,
      'n_lines': n_lines,
      'required_symbols_missing': missing,
      'ok': bool(sha) and not missing and (n_lines or 0) > 100,
      # Canonical full-file hash after 2026-10-01 restore (stash match).
      'expected_sha256_full': (
          '1a3f6c494ed8bee8ea63fb66d7ff2b3e3f00cb3bf13555a3cb877749d1202f9e'),
  }


def assert_forward_process_utils_ok(*, require_expected_sha: bool = False) -> dict:
  """Raise if utils.py looks like the 45-line git stub."""
  fp = forward_process_utils_fingerprint()
  if not fp['ok']:
    raise RuntimeError(
        'forward_process/utils.py is incomplete or missing Unif helpers '
        f"(n_lines={fp['n_lines']}, missing={fp['required_symbols_missing']}, "
        f"sha256={fp['sha256']}). Refusing eval — likely a clean checkout of "
        'the 45-line HEAD stub. Restore the full file or pull the baseline tag.')
  if require_expected_sha and fp['sha256'] != fp['expected_sha256_full']:
    raise RuntimeError(
        f"utils.py sha256={fp['sha256']} != expected "
        f"{fp['expected_sha256_full']} (baseline full blob)")
  return fp


def code_fingerprint_header() -> dict:
  """Small dict for embedding in every eval / probe JSON."""
  utils_fp = forward_process_utils_fingerprint()
  return {
      'forward_process_utils': utils_fp,
      'critical_files': {
          rel: file_sha256(_pkg_root() / rel)
          for rel in _CRITICAL
          if (_pkg_root() / rel).is_file()
      },
  }
