"""Early stopping callbacks for hyperparameter tuning.

Implements collapse detection for JEPA training and performance-based
early termination for efficient hyperparameter search.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback


class CollapseDetectionCallback(Callback):
    """Detect and terminate runs with representation collapse.
    
    Monitors latent space statistics and stops training if:
    - Latent standard deviation falls below threshold (collapse)
    - Loss exceeds baseline by large factor (divergence)
    - Training metrics show no improvement
    
    Following I-JEPA and VICReg best practices for collapse prevention.
    """
    
    def __init__(
        self,
        min_steps: int = 5000,
        std_threshold: float = 0.1,
        loss_threshold: float = 10.0,
        patience: int = 5,
        check_interval: int = 500,
        save_diagnostics: bool = True,
    ):
        """Initialize collapse detection callback.
        
        Args:
            min_steps: Minimum steps before early stopping can trigger
            std_threshold: Stop if latent std falls below this value
            loss_threshold: Stop if loss exceeds baseline * this factor
            patience: Number of consecutive collapse detections before stopping
            check_interval: Steps between collapse checks
            save_diagnostics: Whether to save diagnostic metrics to file
        """
        super().__init__()
        self.min_steps = min_steps
        self.std_threshold = std_threshold
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.check_interval = check_interval
        self.save_diagnostics = save_diagnostics
        
        # State
        self._baseline_loss: Optional[float] = None
        self._collapse_count = 0
        self._diagnostics = []
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Check for collapse after each batch."""
        step = trainer.global_step
        
        # Skip early steps and non-check intervals
        if step < self.min_steps or step % self.check_interval != 0:
            return
        
        # Extract metrics from logged values
        metrics = trainer.callback_metrics
        
        # Get latent statistics
        latent_std = metrics.get('latent/pred_std_mean', None)
        latent_loss = metrics.get('latent/pred_loss', None)
        cosine_sim = metrics.get('latent/cosine_similarity', None)
        pairwise_dist = metrics.get('latent/pred_pairwise_dist', None)
        
        # Initialize baseline loss
        if self._baseline_loss is None and latent_loss is not None:
            if isinstance(latent_loss, torch.Tensor):
                latent_loss = latent_loss.item()
            self._baseline_loss = latent_loss
        
        # Check collapse conditions
        is_collapsed = False
        collapse_reason = None
        
        # Check 1: Low standard deviation
        if latent_std is not None:
            if isinstance(latent_std, torch.Tensor):
                latent_std = latent_std.item()
            if latent_std < self.std_threshold:
                is_collapsed = True
                collapse_reason = f"latent_std ({latent_std:.4f}) < threshold ({self.std_threshold})"
        
        # Check 2: Loss divergence
        if latent_loss is not None and self._baseline_loss is not None:
            if isinstance(latent_loss, torch.Tensor):
                latent_loss = latent_loss.item()
            if latent_loss > self._baseline_loss * self.loss_threshold:
                is_collapsed = True
                collapse_reason = f"loss ({latent_loss:.4f}) > {self.loss_threshold}x baseline ({self._baseline_loss:.4f})"
        
        # Check 3: Very low pairwise distance (all representations identical)
        if pairwise_dist is not None:
            if isinstance(pairwise_dist, torch.Tensor):
                pairwise_dist = pairwise_dist.item()
            if pairwise_dist < 0.01:  # Nearly identical representations
                is_collapsed = True
                collapse_reason = f"pairwise_dist ({pairwise_dist:.4f}) nearly zero"
        
        # Track diagnostics
        diagnostic = {
            "step": step,
            "latent_std": float(latent_std) if latent_std is not None else None,
            "latent_loss": float(latent_loss) if latent_loss is not None else None,
            "cosine_sim": float(cosine_sim) if cosine_sim is not None else None,
            "pairwise_dist": float(pairwise_dist) if pairwise_dist is not None else None,
            "is_collapsed": is_collapsed,
            "collapse_reason": collapse_reason
        }
        self._diagnostics.append(diagnostic)
        
        # Update collapse counter
        if is_collapsed:
            self._collapse_count += 1
            print(f"\n⚠️  Collapse detected at step {step}: {collapse_reason}")
        else:
            self._collapse_count = 0
        
        # Stop if consecutive collapses exceed patience
        if self._collapse_count >= self.patience:
            print(f"\n🛑 Stopping training: {self.patience} consecutive collapse detections")
            trainer.should_stop = True
            
            # Log early stop reason
            pl_module.log("early_stop/reason", 1.0)  # 1.0 = collapse
            pl_module.log("early_stop/step", float(step))
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Save diagnostics at end of training."""
        if self.save_diagnostics:
            output_dir = Path(trainer.log_dir or ".")
            diagnostics_path = output_dir / "collapse_diagnostics.json"
            
            summary = {
                "total_checks": len(self._diagnostics),
                "collapse_detections": sum(1 for d in self._diagnostics if d["is_collapsed"]),
                "final_collapse_count": self._collapse_count,
                "baseline_loss": self._baseline_loss,
                "was_early_stopped": trainer.should_stop and self._collapse_count >= self.patience,
                "diagnostics": self._diagnostics
            }
            
            with open(diagnostics_path, 'w') as f:
                json.dump(summary, f, indent=2)


class PerformanceEarlyStoppingCallback(Callback):
    """Early stopping based on validation performance.
    
    For Stage 2 decoder training, stops if accuracy doesn't improve.
    """
    
    def __init__(
        self,
        monitor: str = "decoder/accuracy",
        min_delta: float = 0.001,
        patience: int = 10,
        mode: str = "max",
        min_steps: int = 5000,
        check_interval: int = 1000,
    ):
        """Initialize performance early stopping.
        
        Args:
            monitor: Metric to monitor
            min_delta: Minimum improvement to reset patience
            patience: Number of checks without improvement before stopping
            mode: 'min' or 'max'
            min_steps: Minimum steps before stopping can trigger
            check_interval: Steps between checks
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.min_steps = min_steps
        self.check_interval = check_interval
        
        self._best_value: Optional[float] = None
        self._no_improve_count = 0
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        step = trainer.global_step
        
        if step < self.min_steps or step % self.check_interval != 0:
            return
        
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        
        if isinstance(current, torch.Tensor):
            current = current.item()
        
        # Initialize best value
        if self._best_value is None:
            self._best_value = current
            return
        
        # Check for improvement
        if self.mode == "max":
            improved = current > self._best_value + self.min_delta
        else:
            improved = current < self._best_value - self.min_delta
        
        if improved:
            self._best_value = current
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        
        # Stop if no improvement
        if self._no_improve_count >= self.patience:
            print(f"\n🛑 Stopping: No improvement in {self.monitor} for {self.patience} checks")
            trainer.should_stop = True
            pl_module.log("early_stop/reason", 2.0)  # 2.0 = no improvement
            pl_module.log("early_stop/step", float(step))


class MetricLoggingCallback(Callback):
    """Log comprehensive metrics for hyperparameter analysis."""
    
    def __init__(
        self,
        log_interval: int = 100,
        checkpoint_steps: list = None,
    ):
        """Initialize metric logging callback.
        
        Args:
            log_interval: Steps between metric logging
            checkpoint_steps: Steps at which to save detailed metrics
        """
        super().__init__()
        self.log_interval = log_interval
        self.checkpoint_steps = checkpoint_steps or [10000, 50000, 100000, 200000]
        self._metrics_history = []
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        step = trainer.global_step
        
        if step % self.log_interval == 0:
            metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in trainer.callback_metrics.items()}
            metrics["step"] = step
            self._metrics_history.append(metrics)
        
        # Save checkpoint metrics
        if step in self.checkpoint_steps:
            self._save_checkpoint_metrics(trainer, step)
    
    def _save_checkpoint_metrics(self, trainer: L.Trainer, step: int):
        """Save detailed metrics at checkpoint."""
        output_dir = Path(trainer.log_dir or ".")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_metrics = {
            "step": step,
            "metrics": {k: v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in trainer.callback_metrics.items()},
            "history_summary": self._compute_history_summary()
        }
        
        with open(output_dir / f"metrics_step_{step}.json", 'w') as f:
            json.dump(checkpoint_metrics, f, indent=2)
    
    def _compute_history_summary(self) -> Dict[str, Any]:
        """Compute summary statistics from metrics history."""
        if not self._metrics_history:
            return {}
        
        import numpy as np
        
        summary = {}
        keys = set()
        for m in self._metrics_history:
            keys.update(m.keys())
        
        for key in keys:
            if key == "step":
                continue
            values = [m.get(key) for m in self._metrics_history if m.get(key) is not None]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "last": values[-1]
                }
        
        return summary
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Save final metrics."""
        output_dir = Path(trainer.log_dir or ".")
        
        with open(output_dir / "metrics_history.json", 'w') as f:
            json.dump(self._metrics_history, f, indent=2)
        
        # Save final metrics for experiment tracker
        if self._metrics_history:
            final_metrics = self._metrics_history[-1].copy()
            final_metrics["summary"] = self._compute_history_summary()
            
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(final_metrics, f, indent=2)

