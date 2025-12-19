"""DDP static graph callback for gradient checkpointing compatibility.

This callback sets a static graph on DistributedDataParallel models when gradient
checkpointing is enabled. This is required to avoid "parameter marked ready twice"
errors that occur when using reentrant gradient checkpointing with DDP.

References:
- PyTorch DDP documentation on static graph optimization
- Gradient checkpointing with DDP best practices
"""

import logging
from typing import Optional

import lightning.pytorch as L
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class DDPStaticGraphCallback(L.Callback):
    """Set static graph on DDP model for gradient checkpointing compatibility.
    
    This callback automatically detects when:
    1. DDP is being used (via Lightning strategy)
    2. Gradient checkpointing is enabled on the model
    3. Sets static graph on the DDP-wrapped model to prevent parameter marking errors
    
    The static graph optimization tells DDP that the model's computational graph
    structure doesn't change between iterations, which is required for compatibility
    with reentrant gradient checkpointing.
    
    Args:
        enabled: Whether to enable this callback (default: True)
        log_level: Logging level for status messages (default: INFO)
    """
    
    def __init__(self, enabled: bool = True, log_level: int = logging.INFO):
        super().__init__()
        self.enabled = enabled
        self.log_level = log_level
        self._static_graph_set = False
    
    def _has_gradient_checkpointing(self, pl_module: L.LightningModule) -> bool:
        """Check if gradient checkpointing is enabled on the model.
        
        Args:
            pl_module: Lightning module to check
            
        Returns:
            True if gradient checkpointing is enabled, False otherwise
        """
        # Check if model has a backbone attribute (common pattern in this codebase)
        if hasattr(pl_module, 'backbone'):
            backbone = pl_module.backbone
            if hasattr(backbone, 'gradient_checkpointing'):
                return getattr(backbone, 'gradient_checkpointing', False)
        
        # Check direct attribute on module
        if hasattr(pl_module, 'gradient_checkpointing'):
            return getattr(pl_module, 'gradient_checkpointing', False)
        
        # Check if model itself has the attribute
        if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'gradient_checkpointing'):
            return getattr(pl_module.model, 'gradient_checkpointing', False)
        
        return False
    
    def _is_ddp_strategy(self, trainer: L.Trainer) -> bool:
        """Check if trainer is using DDP strategy.
        
        Args:
            trainer: Lightning trainer to check
            
        Returns:
            True if using DDP, False otherwise
        """
        strategy = trainer.strategy
        strategy_name = strategy.__class__.__name__.lower()
        
        # Check for DDP strategies (Lightning uses various DDP strategy names)
        ddp_strategies = ['ddp', 'ddpspawn', 'ddp_find_unused_parameters', 'ddp_spawn']
        return any(ddp_name in strategy_name for ddp_name in ddp_strategies)
    
    def _get_ddp_model(self, trainer: L.Trainer) -> Optional[nn.Module]:
        """Get the DDP-wrapped model from trainer.
        
        Args:
            trainer: Lightning trainer
            
        Returns:
            DDP-wrapped model if available, None otherwise
        """
        # Lightning DDP strategies store the wrapped model in strategy.model
        if not hasattr(trainer.strategy, 'model'):
            return None
        
        model = trainer.strategy.model
        
        # Check if it's a DDP model directly
        if isinstance(model, nn.parallel.DistributedDataParallel):
            return model
        
        # Some Lightning versions wrap it differently - check if it's wrapped
        # by looking for DDP-specific attributes
        if hasattr(model, 'module') and hasattr(model, 'process_group'):
            # This looks like a DDP model even if isinstance check fails
            # Try to cast or return as-is
            return model
        
        return None
    
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Set static graph on DDP model when training starts.
        
        This is called after the model is wrapped by DDP but before training begins.
        
        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
        """
        if not self.enabled:
            return
        
        # Check if DDP is being used
        if not self._is_ddp_strategy(trainer):
            logger.log(
                self.log_level,
                "DDPStaticGraphCallback: Not using DDP strategy, skipping static graph setup"
            )
            return
        
        # Check if gradient checkpointing is enabled
        if not self._has_gradient_checkpointing(pl_module):
            logger.log(
                self.log_level,
                "DDPStaticGraphCallback: Gradient checkpointing not enabled, skipping static graph setup"
            )
            return
        
        # Get DDP model
        ddp_model = self._get_ddp_model(trainer)
        if ddp_model is None:
            # Try alternative access patterns for different Lightning versions
            if hasattr(trainer.strategy, '_model'):
                ddp_model = trainer.strategy._model
            elif hasattr(trainer.strategy, '_lightning_module'):
                # Sometimes the model is stored differently
                lightning_module = trainer.strategy._lightning_module
                if hasattr(lightning_module, '_ddp_model'):
                    ddp_model = lightning_module._ddp_model
            
            if ddp_model is None or not isinstance(ddp_model, nn.parallel.DistributedDataParallel):
                logger.warning(
                    "DDPStaticGraphCallback: Could not find DDP-wrapped model, "
                    "static graph will not be set. This may cause errors with gradient checkpointing. "
                    f"Strategy type: {type(trainer.strategy).__name__}"
                )
                return
        
        # Set static graph if method exists
        if hasattr(ddp_model, '_set_static_graph'):
            try:
                ddp_model._set_static_graph()
                self._static_graph_set = True
                logger.log(
                    self.log_level,
                    "DDPStaticGraphCallback: Successfully set static graph on DDP model "
                    "for gradient checkpointing compatibility"
                )
            except Exception as e:
                logger.warning(
                    f"DDPStaticGraphCallback: Failed to set static graph: {e}. "
                    "Training may fail with gradient checkpointing enabled."
                )
        else:
            logger.warning(
                "DDPStaticGraphCallback: DDP model does not have _set_static_graph method. "
                "This may indicate an older PyTorch version (<1.12) or a different DDP implementation. "
                "Consider upgrading PyTorch or disabling gradient checkpointing."
            )

