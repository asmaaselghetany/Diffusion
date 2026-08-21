"""Checkpoint helpers for Lightning ModelCheckpoint edge cases.

1. ``SaveLastOnTrainEnd`` — Lightning 2.5's ModelCheckpoint.on_train_end is a
   no-op. With ``every_n_train_steps=N``, a run that hits ``max_steps`` where
   ``max_steps % N != 0`` never writes a final checkpoint.

2. ``SeedCheckpointResumeStep`` — ModelCheckpoint does **not** persist
   ``_last_global_step_saved`` in ``state_dict``. After resume at a multiple of
   ``every_n_train_steps``, the first ``on_train_batch_end`` re-saves that same
   step (periodic + last → multi‑GB Lustre rewrite before any training).
"""

from __future__ import annotations

import logging

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)


class SeedCheckpointResumeStep(Callback):
  """Prevent immediate re-checkpoint after resume at a save boundary."""

  def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
    del pl_module
    step = int(trainer.global_step)
    if step <= 0:
      return
    seeded = 0
    for cb in trainer.callbacks:
      if not isinstance(cb, ModelCheckpoint):
        continue
      prev = int(getattr(cb, "_last_global_step_saved", 0) or 0)
      if prev < step:
        cb._last_global_step_saved = step
        seeded += 1
    if seeded and trainer.global_rank == 0:
      logger.info(
        "SeedCheckpointResumeStep: set _last_global_step_saved=%s on %s "
        "ModelCheckpoint(s) (avoids re-saving current step on resume)",
        step,
        seeded,
      )


class SaveLastOnTrainEnd(Callback):
  def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    if trainer.global_rank != 0:
      return
    saved = False
    for cb in trainer.callbacks:
      if not isinstance(cb, ModelCheckpoint) or not cb.save_last:
        continue
      if cb._should_skip_saving_checkpoint(trainer):
        continue
      monitor_candidates = cb._monitor_candidates(trainer)
      cb._save_last_checkpoint(trainer, monitor_candidates)
      saved = True
      logger.info(
        "SaveLastOnTrainEnd: wrote last ckpt at global_step=%s → %s",
        trainer.global_step,
        cb.last_model_path,
      )
    if not saved:
      logger.warning(
        "SaveLastOnTrainEnd: no ModelCheckpoint(save_last) wrote at step=%s",
        trainer.global_step,
      )
