"""Keep only the newest N periodic ``0-*.ckpt`` / ``1-*.ckpt`` files.

``ModelCheckpoint(every_n_train_steps=…, save_top_k=-1)`` otherwise retains
every periodic (~24GB each for Qwen-1.5B). Combined with ``best`` + ``last``
that can exceed project quota mid-run (ENOSPC on commit).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)

_PERIODIC_RE = re.compile(r'^(\d+)-(\d+)\.ckpt$')


def prune_periodic_checkpoints(
    ckpt_dir: str | Path,
    *,
    keep_newest: int = 2,
) -> list[Path]:
  """Delete older periodic step ckpts; keep ``last``/``best`` untouched.

  Returns paths that were removed.
  """
  root = Path(ckpt_dir)
  if not root.is_dir():
    return []
  periodics: list[tuple[int, int, Path]] = []
  for path in root.iterdir():
    if not path.is_file():
      continue
    m = _PERIODIC_RE.match(path.name)
    if not m:
      continue
    epoch, step = int(m.group(1)), int(m.group(2))
    periodics.append((step, epoch, path))
  periodics.sort(key=lambda t: (t[0], t[1]))
  if keep_newest < 0:
    return []
  victims = periodics[:-keep_newest] if keep_newest else periodics
  removed: list[Path] = []
  for _, _, path in victims:
    try:
      path.unlink()
      removed.append(path)
    except OSError as exc:
      logger.warning('prune: could not remove %s: %s', path, exc)
  return removed


class PrunePeriodicCheckpoints(Callback):
  """After each checkpoint save, keep only ``keep_newest`` periodics."""

  def __init__(
      self,
      dirpath: str,
      keep_newest: int = 2,
  ) -> None:
    super().__init__()
    self.dirpath = dirpath
    self.keep_newest = int(keep_newest)

  def _prune(self) -> None:
    removed = prune_periodic_checkpoints(
        self.dirpath, keep_newest=self.keep_newest)
    if removed:
      logger.info(
          'Pruned %d periodic checkpoint(s); kept newest %d under %s',
          len(removed), self.keep_newest, self.dirpath)

  def on_train_batch_end(
      self,
      trainer: Trainer,
      pl_module: LightningModule,
      outputs,
      batch,
      batch_idx: int,
  ) -> None:
    del pl_module, outputs, batch, batch_idx
    # Rank 0 only — other ranks may not see the just-written file yet.
    if not trainer.is_global_zero:
      return
    self._prune()

  def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    del pl_module
    if trainer.is_global_zero:
      self._prune()
