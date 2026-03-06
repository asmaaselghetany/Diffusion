"""Latent-space evaluation callback for JEPA models."""

import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer

from ..forward_process import BlockAbsorbingForwardProcess


class LatentEvalCallback(Callback):
    """Evaluate latent-space metrics periodically.
    
    Computes on validation batches:
    - Masked MSE between predicted and teacher latents
    - Cosine similarity
    - Latent norm and variance statistics
    
    Args:
        eval_frequency: Evaluate every N training steps
        num_batches: Number of validation batches to evaluate
        sampling_eps: Epsilon for timestep sampling
    """
    
    def __init__(self, eval_frequency: int = 1000, num_batches: int = 10, sampling_eps: float = 1e-3):
        super().__init__()
        self.eval_frequency = eval_frequency
        self.num_batches = num_batches
        self.sampling_eps = sampling_eps
    
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):
        if trainer.global_step % self.eval_frequency != 0 or trainer.global_step == 0:
            return
        
        pl_module.eval()
        try:
            metrics = self._run_evaluation(trainer, pl_module)
            for key, value in metrics.items():
                pl_module.log(key, value, prog_bar=False, logger=True, sync_dist=True)
        finally:
            pl_module.train()
    
    def _run_evaluation(self, trainer: Trainer, pl_module: LightningModule) -> dict:
        if not hasattr(pl_module, 'backbone') or not hasattr(pl_module.backbone, 'encode_student'):
            return {}
        
        forward_process = pl_module._forward_process
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return {}
        
        metrics = {"latent_eval/mse": [], "latent_eval/cosine_sim": [], 
                   "latent_eval/latent_norm": [], "latent_eval/latent_var": []}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= self.num_batches:
                    break
                
                input_sequence = batch["input_ids"].to(pl_module.device)
                B, L = input_sequence.shape
                
                t = torch.rand(B, device=pl_module.device).clamp(self.sampling_eps, 1.0 - self.sampling_eps)
                
                if isinstance(forward_process, BlockAbsorbingForwardProcess):
                    x_t, _, t = forward_process(input_sequence, t)
                    t = t.mean(dim=1)
                else:
                    x_t, _ = forward_process(input_sequence, t)
                
                z_t = pl_module.backbone.encode_student(x_t, t)
                z_0 = pl_module.backbone.encode_teacher(input_sequence, torch.zeros_like(t))
                z_hat_0 = pl_module.backbone.predict_latent(z_t, t)
                
                mask_indices = (x_t == pl_module.mask_id)
                num_masked = mask_indices.sum().clamp(min=1)
                
                mse = ((z_hat_0 - z_0) ** 2).sum(dim=-1)
                metrics["latent_eval/mse"].append((mse * mask_indices.float()).sum().item() / num_masked.item())
                
                cosine_sim = (F.normalize(z_hat_0, p=2, dim=-1) * F.normalize(z_0, p=2, dim=-1)).sum(dim=-1)
                metrics["latent_eval/cosine_sim"].append((cosine_sim * mask_indices.float()).sum().item() / num_masked.item())
                
                metrics["latent_eval/latent_norm"].append(z_hat_0.norm(p=2, dim=-1).mean().item())
                
                z_flat = z_hat_0[mask_indices]
                if z_flat.numel() > 0:
                    metrics["latent_eval/latent_var"].append(z_flat.var(dim=0).mean().item())
        
        return {k: sum(v) / len(v) for k, v in metrics.items() if v}
