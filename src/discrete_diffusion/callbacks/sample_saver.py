"""Periodic sample saving hook for discrete diffusion models.

Generates a small batch of samples every ``every_n_steps`` training steps,
saves decoded text to disk and (optionally) logs a W&B text table plus
token-entropy scalar.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


class SampleSaver(L.Callback):
  """Save generated tokens every ``every_n_steps`` during training.

  When ``log_to_wandb`` is *True* (the default) and a W&B logger is
  attached, a text table and the token entropy are also logged so
  generation quality can be tracked on the dashboard without opening
  JSON files.
  """

  def __init__(
      self,
      enabled: bool = False,
      every_n_steps: int = 1000,
      num_samples: Optional[int] = None,
      num_steps: Optional[int] = None,
      save_dir: str = './samples/',
      filename_template: str = 'step_{global_step}.json',
      log_to_wandb: bool = True) -> None:
    super().__init__()
    if every_n_steps <= 0:
      raise ValueError('every_n_steps must be positive')

    self.enabled = enabled
    self.every_n_steps = every_n_steps
    self.num_samples = num_samples
    self.num_steps = num_steps
    self.save_dir = Path(save_dir)
    self.filename_template = filename_template
    self.log_to_wandb = log_to_wandb

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    del outputs, batch, batch_idx
    if not self.enabled or not trainer.is_global_zero:
      return

    global_step = max(1, trainer.global_step)
    if global_step % self.every_n_steps != 0:
      return

    try:
      samples = pl_module.generate_samples(
        num_samples=self._resolve_num_samples(pl_module),
        num_steps=self._resolve_num_steps(pl_module))
    except Exception as exc:
      log.warning("SampleSaver: generation failed at step %d: %s", global_step, exc)
      return
    samples = samples.detach().cpu()

    text_samples = pl_module.tokenizer.batch_decode(samples.tolist())
    entropy = self._mean_entropy(samples)

    save_path = self._build_save_path(global_step)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(
      text=text_samples,
      entropy=entropy,
      config=OmegaConf.to_container(pl_module.config, resolve=True),
    )
    with open(save_path, 'w', encoding='utf-8') as fp:
      json.dump(metadata, fp, indent=2)

    pl_module.log('samples/entropy', entropy, on_step=True, on_epoch=False)

    if self.log_to_wandb:
      self._log_wandb(trainer, text_samples, global_step)

  @staticmethod
  def _log_wandb(trainer, text_samples, global_step):
    logger = trainer.logger
    if logger is None:
      return
    if hasattr(logger, 'log_table'):
      try:
        logger.log_table(
          key=f'samples@step{global_step}',
          columns=['idx', 'text'],
          data=[[i, t] for i, t in enumerate(text_samples)],
        )
      except Exception:
        pass

  def _mean_entropy(self, samples: torch.Tensor) -> float:
    if samples.numel() == 0:
      return 0.0
    entropies = []
    for sample in samples.unbind(0):
      _, counts = torch.unique(sample, return_counts=True, sorted=False)
      probs = counts.float() / counts.sum()
      entropies.append(float(torch.special.entr(probs).sum()))
    return float(sum(entropies) / len(entropies))

  def _build_save_path(self, global_step: int) -> Path:
    filename = self.filename_template.format(global_step=global_step)
    return self.save_dir / filename

  def _resolve_num_samples(self, pl_module) -> int:
    if self.num_samples is not None:
      return self.num_samples
    batch_size = getattr(pl_module.config.loader, 'eval_batch_size', None)
    if batch_size is None:
      raise ValueError('Could not infer num_samples for SampleSaver')
    return batch_size

  def _resolve_num_steps(self, pl_module) -> int:
    if self.num_steps is not None:
      return self.num_steps
    steps = getattr(pl_module.config.sampling, 'steps', None)
    if steps is None:
      raise ValueError('Could not infer sampling steps for SampleSaver')
    return steps
