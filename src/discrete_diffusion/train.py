"""Public training API for discrete diffusion models."""

import os

import hydra
import lightning as L
import omegaconf
import torch

from .data import get_dataloaders, get_tokenizer
from . import utils
from .callbacks.ddp_static_graph import DDPStaticGraphCallback


def train(config):
  """Main training API.
  
  Args:
    config: Hydra DictConfig or config object with training parameters.
    
  Returns:
    None. Model checkpoints are saved according to config.checkpointing.
  """
  # Set matmul precision to 'high' (TF32) to match FlexMDM
  torch.set_float32_matmul_precision("high")
  
  logger = utils.get_logger(__name__)
  logger.info('Starting Training.')
  
  tokenizer = get_tokenizer(config)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  
  # Load dataloaders (data should be pre-cached for multi-node)
  train_ds, valid_ds = get_dataloaders(config, tokenizer)
  
  # WandB logger
  wandb_logger = L.pytorch.loggers.WandbLogger(
    config=omegaconf.OmegaConf.to_object(config), **config.wandb
  ) if config.get('wandb', None) is not None else None

  # Resume checkpoint path
  ckpt_path = config.checkpointing.resume_ckpt_path if (
    config.checkpointing.resume_from_ckpt and 
    config.checkpointing.resume_ckpt_path is not None and 
    utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
  ) else None

  # Lightning callbacks
  callbacks = [hydra.utils.instantiate(cb) for _, cb in config.callbacks.items()] if 'callbacks' in config else []

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = algo_cls.load_from_checkpoint(
      config.training.finetune_path, tokenizer=tokenizer, config=config, strict=False)
  else:
    model = algo_cls(config, tokenizer=tokenizer)

  # Torch compile if enabled
  if omegaconf.OmegaConf.select(config, 'training.torch_compile', default=False):
    logger.info('Compiling LightningModule with torch.compile.')
    model = torch.compile(model)

  if config.training.get('fault_tolerant', False):
    os.environ.setdefault('PL_FAULT_TOLERANT_TRAINING', '1')

  use_dist = config.trainer.get('num_nodes', 1) > 1 or config.trainer.get('devices', 1) > 1
  
  # Check if gradient checkpointing is enabled (check both model and config)
  gradient_checkpointing_enabled = False
  # First check the instantiated model
  if hasattr(model, 'backbone') and hasattr(model.backbone, 'gradient_checkpointing'):
    gradient_checkpointing_enabled = getattr(model.backbone, 'gradient_checkpointing', False)
  elif hasattr(model, 'gradient_checkpointing'):
    gradient_checkpointing_enabled = getattr(model, 'gradient_checkpointing', False)
  # Fallback to config if model doesn't have the attribute yet
  if not gradient_checkpointing_enabled and hasattr(config, 'model'):
    if hasattr(config.model, 'gradient_checkpointing'):
      gradient_checkpointing_enabled = getattr(config.model, 'gradient_checkpointing', False)
  
  # Determine DDP strategy
  # When using static graph (for gradient checkpointing), we can use regular 'ddp'
  # which is more efficient than 'ddp_find_unused_parameters_true'
  if config.get('strategy', None) is not None:
    strategy = hydra.utils.instantiate(config.strategy)
  else:
    if use_dist:
      # Use regular DDP when gradient checkpointing is enabled (static graph will be set)
      # Otherwise use find_unused_parameters for safety
      if gradient_checkpointing_enabled:
        strategy = 'ddp'
        logger.info(
          'Gradient checkpointing detected with DDP. Using "ddp" strategy. '
          'Static graph will be set automatically for compatibility.'
        )
      else:
        strategy = 'ddp_find_unused_parameters_true'
    else:
      strategy = 'auto'
  
  # Automatically add DDP static graph callback if needed
  # Check if callback is already in the list (from config or manually added)
  has_ddp_callback = any(
    isinstance(cb, DDPStaticGraphCallback) for cb in callbacks
  )
  
  if use_dist and gradient_checkpointing_enabled and not has_ddp_callback:
    ddp_callback = DDPStaticGraphCallback(enabled=True)
    callbacks.append(ddp_callback)
    logger.info(
      'Automatically added DDPStaticGraphCallback for gradient checkpointing compatibility.'
    )
  
  trainer = L.Trainer(
    **config.trainer, default_root_dir=os.getcwd(), callbacks=callbacks,
    strategy=strategy, logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)