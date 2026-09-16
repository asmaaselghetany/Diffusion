"""Public training API for discrete diffusion models."""

import os

import hydra
import lightning as L
import omegaconf
import torch

from .data import get_dataloaders, get_tokenizer
from . import utils


def register_config_resolvers():
  """Register OmegaConf resolvers used in Hydra YAML (also needed for script compose)."""
  import functools
  import operator

  def _mul(*args):
    return functools.reduce(operator.mul, args) if args else 1

  def _default_device():
    if not torch.cuda.is_available():
      raise RuntimeError(
          'CUDA is required for training. Run on a GPU node (e.g. via Slurm).')
    return 'cuda:0'

  for name, fn in [
      ('cwd', os.getcwd),
      ('device_count', lambda: torch.cuda.device_count()),
      ('default_device', _default_device),
      ('div_up', lambda x, y: (x + y - 1) // y),
      ('mul', _mul),
      ('sub', lambda x, y: x - y),
  ]:
    if not omegaconf.OmegaConf.has_resolver(name):
      omegaconf.OmegaConf.register_new_resolver(name, fn)


def _resolve_accelerator(config) -> str:
  accel = omegaconf.OmegaConf.select(config, 'trainer.accelerator', default='auto')
  if accel not in ('auto', 'cuda', 'gpu'):
    if accel in ('cpu', 'mps'):
      raise ValueError(
          f'trainer.accelerator={accel!r} is not supported; use cuda on cluster GPUs.')
    return accel
  if not torch.cuda.is_available():
    raise RuntimeError(
        'CUDA is required for training. Run on a GPU node (e.g. via Slurm).')
  return 'cuda'


def _strategy_device(accel: str) -> str:
  if accel == 'cuda':
    return 'cuda:0'
  return accel


def _align_strategy_device(config, accel: str) -> None:
  if not omegaconf.OmegaConf.is_config(config.get('strategy', None)):
    return
  if 'device' not in config.strategy:
    return
  omegaconf.OmegaConf.set_struct(config.strategy, False)
  config.strategy.device = _strategy_device(accel)
  omegaconf.OmegaConf.set_struct(config.strategy, True)


def train(config):
  """Main training API.
  
  Args:
    config: Hydra DictConfig or config object with training parameters.
    
  Returns:
    None. Model checkpoints are saved according to config.checkpointing.
  """
  register_config_resolvers()
  # Set matmul precision to 'high' (TF32) to match FlexMDM
  torch.set_float32_matmul_precision("high")
  
  logger = utils.get_logger(__name__)
  logger.info('Starting Training.')
  
  tokenizer = get_tokenizer(config)
  # Match embed table when resuming / finetuning an older (shrunk) ckpt.
  for ckpt_candidate in (
      getattr(config.checkpointing, 'resume_ckpt_path', None)
      if getattr(config.checkpointing, 'resume_from_ckpt', False) else None,
      getattr(config.training, 'finetune_path', None) or None,
  ):
    if not ckpt_candidate or not utils.fsspec_exists(ckpt_candidate):
      continue
    try:
      from discrete_diffusion.data.conversion_baseline import (
          apply_checkpoint_embed_vocab_size,
          checkpoint_embed_vocab_size,
      )
      peek = torch.load(ckpt_candidate, map_location='cpu', weights_only=False)
      embed_v = checkpoint_embed_vocab_size(peek)
      if embed_v is not None:
        apply_checkpoint_embed_vocab_size(tokenizer, embed_v)
        logger.info(
            'Matched tokenizer embed vocab_size=%s from %s',
            embed_v, ckpt_candidate)
      break
    except Exception as exc:
      logger.warning('Could not peek ckpt vocab from %s: %s', ckpt_candidate, exc)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  
  accel = _resolve_accelerator(config)
  omegaconf.OmegaConf.set_struct(config.trainer, False)
  config.trainer.accelerator = accel
  omegaconf.OmegaConf.set_struct(config.trainer, True)
  _align_strategy_device(config, accel)

  # Ensure dataset processing happens on rank 0 first
  fabric = L.Fabric(num_nodes=config.trainer.num_nodes,
                    devices=config.trainer.devices,
                    accelerator=accel)
  fabric.launch()
  with fabric.rank_zero_first():
    train_ds, valid_ds = get_dataloaders(config, tokenizer)
  fabric.barrier()
  del fabric
  
  # WandB only on global rank 0 under external DDP (srun). Creating WandbLogger
  # on all ranks blocks rank0 in wandb.init while rank1 waits on NCCL.
  # Non-zero ranks still need *some* logger — LearningRateMonitor crashes with
  # "Trainer that has no logger" if logger=False.
  global_rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
  want_wandb = omegaconf.OmegaConf.select(config, 'wandb') is not None
  if want_wandb and global_rank == 0:
    train_logger = L.pytorch.loggers.WandbLogger(
        config=omegaconf.OmegaConf.to_object(config), **config.wandb)
  elif want_wandb:
    train_logger = L.pytorch.loggers.CSVLogger(
        save_dir=os.getcwd(), name='lightning_logs', version=f'rank{global_rank}')
  else:
    train_logger = False

  # Resume checkpoint path
  ckpt_path = config.checkpointing.resume_ckpt_path if (
    config.checkpointing.resume_from_ckpt and 
    config.checkpointing.resume_ckpt_path is not None and 
    utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
  ) else None

  # Lightning callbacks
  callbacks_cfg = config.get('callbacks', None)
  if (
      callbacks_cfg is None
      or (omegaconf.OmegaConf.is_list(callbacks_cfg) and len(callbacks_cfg) == 0)
      or (omegaconf.OmegaConf.is_dict(callbacks_cfg) and len(callbacks_cfg) == 0)
  ):
    callbacks = None  # Lightning 2.x rejects callbacks=[] / empty dict
  else:
    callbacks = [
        hydra.utils.instantiate(cb) for _, cb in callbacks_cfg.items()]

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = algo_cls.load_from_checkpoint(
      config.training.finetune_path, tokenizer=tokenizer, config=config)
  else:
    model = algo_cls(config, tokenizer=tokenizer)

  # Torch compile if enabled
  if omegaconf.OmegaConf.select(config, 'training.torch_compile', default=False):
    logger.info('Compiling LightningModule with torch.compile.')
    model = torch.compile(model)

  if config.training.get('fault_tolerant', False):
    os.environ.setdefault('PL_FAULT_TOLERANT_TRAINING', '1')

  trainer = L.Trainer(
    **config.trainer, default_root_dir=os.getcwd(), callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=train_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)