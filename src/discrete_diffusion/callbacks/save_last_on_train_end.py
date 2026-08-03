"""Force ModelCheckpoint save_last at train end.

Lightning 2.5's ModelCheckpoint.on_train_end is a no-op. With
``every_n_train_steps=N``, a run that hits ``max_steps`` where
``max_steps % N != 0`` never writes a final checkpoint — trainer exits 0
with disk still at the previous multiple of N (e.g. 7500 vs save-every-1000
→ last step on disk stays 7000).
"""

from __future__ import annotations

import logging

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)


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
