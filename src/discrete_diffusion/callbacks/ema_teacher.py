"""EMA teacher update callback for latent JEPA models."""

from lightning.pytorch import Callback, LightningModule, Trainer


class EMATeacherCallback(Callback):
    """Update teacher encoder via EMA after each training step.
    
    Args:
        update_frequency: Update EMA every N training steps (default: 1)
    """
    
    def __init__(self, update_frequency: int = 1):
        super().__init__()
        self.update_frequency = update_frequency
    
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        """Update EMA teacher after each training batch."""
        if not hasattr(pl_module, 'backbone') or not hasattr(pl_module.backbone, 'update_ema'):
            return
        if trainer.global_step % self.update_frequency != 0:
            return
        pl_module.backbone.update_ema(trainer.global_step)

