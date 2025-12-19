"""Latent JEPA training algorithm for discrete diffusion.

Supports Stage 1 (latent JEPA training with MSE + VICReg) and 
Stage 2 (decoder-only training with CE loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from . import base as trainer_base
from ..forward_process import BlockAbsorbingForwardProcess


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  num_tokens: torch.FloatTensor


class LatentJEPATrainer(trainer_base.AbsorbingState):
  """Latent JEPA discrete diffusion trainer.
  
  Stage 1: Train student encoder, predictor, readout with MSE + VICReg loss.
  Stage 2: Freeze encoder/predictor, train only readout with CE loss.
  """

  def __init__(self, config, tokenizer):
    # Stage config - must be set before super().__init__ because _get_parameters is called during EMA setup
    self.stage = getattr(config.algo, 'stage', 1)
    
    super().__init__(config, tokenizer)
    
    # JEPA objective params
    self.sampling_eps = getattr(config.algo, 'sampling_eps', 1e-3)
    self.lambda_var = getattr(config.algo, 'lambda_var', 0.1)
    self.lambda_cov = getattr(config.algo, 'lambda_cov', 0.04)
    self.loss_norm = getattr(config.algo, 'loss_norm', 'l2')
    self.mask_only = getattr(config.algo, 'mask_only', True)
    self.redundancy = getattr(config.algo, 'redundancy', 'vicreg').lower()
    self.vicreg_eps = getattr(config.algo, 'vicreg_eps', 1e-4)
    self.normalize_targets = getattr(config.algo, 'normalize_targets', True)
    
    # Noise warmup config
    self.noise_warmup_start = int(getattr(config.algo, 'noise_warmup_start_step', 0))
    self.noise_warmup_end = int(getattr(config.algo, 'noise_warmup_end_step', 0))
    warmup_range = getattr(config.algo, 'noise_warmup_range', [0.15, 0.3])
    self.noise_warmup_range = [float(warmup_range[0]), float(warmup_range[1])]
    
    # Stage 2 specific
    self.latent_source = getattr(config.algo, 'latent_source', 'predicted').lower()
    self.mixed_schedule = getattr(config.algo, 'mixed_schedule', 'linear').lower()
    self.mixed_warmup_steps = int(getattr(config.algo, 'mixed_warmup_steps', 10000))
    self.weight_by_hazard = getattr(config.algo, 'weight_by_hazard', True)
    
    # Target normalization layer for I-JEPA best practice
    if self.normalize_targets:
      self.target_norm = nn.LayerNorm(self.backbone.latent_dim)
    
    # Freeze encoder/predictor for Stage 2
    if self.stage == 2:
      self._freeze_encoder_predictor()

  def _freeze_encoder_predictor(self):
    """Freeze student encoder and predictor for Stage 2 training."""
    for param in self.backbone.student_encoder.parameters():
      param.requires_grad = False
    for param in self.backbone.predictor.parameters():
      param.requires_grad = False

  def _get_parameters(self):
    """Return trainable parameters based on stage."""
    if self.stage == 2:
      return self.backbone.decoder_parameters()
    return self.backbone.optimizable_parameters()

  def optimizer_step(self, *args, **kwargs):
    """Update optimizer and EMA teacher."""
    super().optimizer_step(*args, **kwargs)
    if self.stage == 1:  # Only update EMA during Stage 1
      self.backbone.update_ema(self.global_step)

  def _sample_timesteps(self, batch_size, device):
    """Sample timesteps with optional noise warmup."""
    if (self.noise_warmup_end > 0 and 
        self.noise_warmup_start <= self.global_step < self.noise_warmup_end):
      t_min, t_max = self.noise_warmup_range
      t0 = torch.rand((), device=device)
      t = (t0 + torch.arange(batch_size, device=device, dtype=torch.float32) / max(batch_size, 1)) % 1.0
      t = t_min + t * (t_max - t_min)
      return t.clamp(max(t_min, self.sampling_eps), min(t_max, 1.0 - self.sampling_eps))
    # Standard antithetic sampling
    t0 = torch.rand((), device=device)
    t = (t0 + torch.arange(batch_size, device=device, dtype=torch.float32) / max(batch_size, 1)) % 1.0
    return t.clamp(self.sampling_eps, 1.0 - self.sampling_eps)

  def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
    """Compute JEPA loss based on current stage."""
    if self.stage == 1:
      return self._jepa_loss(x0, valid_tokens)
    return self._decoder_loss(x0, valid_tokens)

  def _jepa_loss(self, input_sequence, valid_tokens):
    """Stage 1: Latent JEPA loss (MSE + VICReg)."""
    B, L = input_sequence.shape
    device = input_sequence.device
    
    t = self._sample_timesteps(B, device)
    
    # Generate masked sequence via forward process
    if isinstance(self._forward_process, BlockAbsorbingForwardProcess):
      x_t, _, t_tok = self._forward_process(input_sequence, t)
    else:
      x_t, _ = self._forward_process(input_sequence, t)
      t_tok = None
    
    # Encode with student and teacher
    z_t = self.backbone.encode_student(x_t, t)
    z_0 = self.backbone.encode_teacher(input_sequence, t)
    
    # Normalize targets (I-JEPA best practice)
    if self.normalize_targets:
      self.target_norm = self.target_norm.to(z_0.device)
      z_0 = self.target_norm(z_0)
    
    # Predict clean latents
    z_hat_0 = self.backbone.predict_latent(z_t, t)
    
    # Identify masked positions
    mask_indices = (x_t == self.mask_id)
    num_masked = mask_indices.sum().clamp(min=1)
    
    # Compute schedule weighting
    if t_tok is not None:
      alpha_t = self.noise.alpha_t(t_tok)
      alpha_prime_t = self.noise.alpha_prime_t(t_tok)
    else:
      alpha_t = self.noise.alpha_t(t)
      alpha_prime_t = self.noise.alpha_prime_t(t)
      alpha_t = alpha_t.view(B, 1).expand(B, L)
      alpha_prime_t = alpha_prime_t.view(B, 1).expand(B, L)
    
    weight = -alpha_prime_t / (1 - alpha_t).clamp(min=self.sampling_eps, max=100)
    
    # Prediction loss
    if self.loss_norm == "l2":
      per_token_loss = ((z_hat_0 - z_0) ** 2).mean(dim=-1)
    else:  # cosine
      z_hat_norm = F.normalize(z_hat_0, p=2, dim=-1)
      z_0_norm = F.normalize(z_0, p=2, dim=-1)
      per_token_loss = 1 - (z_hat_norm * z_0_norm).mean(dim=-1)
    
    if self.mask_only:
      weighted_loss = per_token_loss * mask_indices.float() * weight
    else:
      weighted_loss = per_token_loss * weight
    
    pred_loss = weighted_loss.sum() / num_masked.float()
    
    # Regularization loss
    if self.redundancy == "vicreg":
      reg_loss = self._vicreg_loss(z_hat_0, mask_indices)
    elif self.redundancy == "barlow":
      reg_loss = self._barlow_loss(z_hat_0, mask_indices)
    else:
      reg_loss = torch.tensor(0.0, device=device)
    
    total_loss = pred_loss + reg_loss
    
    # Log metrics with collapse detection
    with torch.no_grad():
      z_hat_norm = F.normalize(z_hat_0, p=2, dim=-1)
      z_0_norm = F.normalize(z_0, p=2, dim=-1)
      cosine_sim = (z_hat_norm * z_0_norm).sum(dim=-1)
      if self.mask_only:
        cosine_sim = (cosine_sim * mask_indices.float()).sum() / num_masked.float()
      else:
        cosine_sim = cosine_sim.mean()
      
      # Collapse detection metrics
      z_hat_masked = z_hat_0[mask_indices] if self.mask_only else z_hat_0.view(-1, z_hat_0.size(-1))
      z_0_masked = z_0[mask_indices] if self.mask_only else z_0.view(-1, z_0.size(-1))
      
      if z_hat_masked.numel() > 0 and z_hat_masked.shape[0] > 1:
        D = z_hat_masked.shape[1]
        # Predicted latents statistics
        z_hat_centered = z_hat_masked - z_hat_masked.mean(dim=0, keepdim=True)
        z_hat_std = z_hat_centered.var(dim=0).sqrt()
        self.log('latent/pred_std_mean', z_hat_std.mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log('latent/pred_std_min', z_hat_std.min(), on_step=True, on_epoch=False, sync_dist=True)
        self.log('latent/pred_std_max', z_hat_std.max(), on_step=True, on_epoch=False, sync_dist=True)
        
        # Teacher latents statistics
        z_0_centered = z_0_masked - z_0_masked.mean(dim=0, keepdim=True)
        z_0_std = z_0_centered.var(dim=0).sqrt()
        self.log('latent/teacher_std_mean', z_0_std.mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log('latent/teacher_std_min', z_0_std.min(), on_step=True, on_epoch=False, sync_dist=True)
        self.log('latent/teacher_std_max', z_0_std.max(), on_step=True, on_epoch=False, sync_dist=True)
        
        # Off-diagonal covariance (redundancy measure)
        if D > 1:
          z_hat_std_norm = z_hat_centered / z_hat_std.clamp(min=1e-4)
          cov_hat = (z_hat_std_norm.T @ z_hat_std_norm) / (z_hat_masked.shape[0] - 1)
          off_diag_mask = ~torch.eye(D, device=device, dtype=torch.bool)
          self.log('latent/pred_cov_offdiag', cov_hat[off_diag_mask].abs().mean(), on_step=True, on_epoch=False, sync_dist=True)
        
        # Pairwise distance (collapse indicator)
        if z_hat_masked.shape[0] >= 2:
          n_samples = min(z_hat_masked.shape[0], 500)
          idx = torch.randperm(z_hat_masked.shape[0], device=device)[:n_samples]
          z_sample = z_hat_masked[idx]
          pdist = torch.cdist(z_sample, z_sample, p=2)
          triu_idx = torch.triu_indices(n_samples, n_samples, offset=1, device=device)
          self.log('latent/pred_pairwise_dist', pdist[triu_idx[0], triu_idx[1]].mean(), on_step=True, on_epoch=False, sync_dist=True)
    
    self.log('latent/pred_loss', pred_loss, on_step=True, on_epoch=False, sync_dist=True)
    self.log('latent/reg_loss', reg_loss, on_step=True, on_epoch=False, sync_dist=True)
    self.log('latent/cosine_similarity', cosine_sim, on_step=True, on_epoch=False, sync_dist=True)
    self.log('latent/weight_mean', weight.mean(), on_step=True, on_epoch=False, sync_dist=True)
    self.log('latent/num_masked', num_masked.float(), on_step=True, on_epoch=False, sync_dist=True)
    
    num_tokens = valid_tokens.sum()
    return Loss(loss=total_loss, nlls=total_loss * num_tokens, num_tokens=num_tokens)

  def _decoder_loss(self, input_sequence, valid_tokens):
    """Stage 2: Decoder CE loss on masked positions."""
    B, L = input_sequence.shape
    device = input_sequence.device
    
    t = self._sample_timesteps(B, device)
    
    # Generate masked sequence
    if isinstance(self._forward_process, BlockAbsorbingForwardProcess):
      x_t, _, t_tok = self._forward_process(input_sequence, t)
      t = t_tok.mean(dim=1) if t_tok is not None else t
    else:
      x_t, _ = self._forward_process(input_sequence, t)
    
    mask_indices = (x_t == self.mask_id)
    num_masked = mask_indices.sum().clamp(min=1)
    
    # Get latents (frozen encoder/predictor)
    with torch.no_grad():
      z_t = self.backbone.encode_student(x_t, t)
      mix_ratio = self._get_mix_ratio()
      
      if mix_ratio >= 1.0:
        z_source = self.backbone.encode_teacher(input_sequence, t)
      elif mix_ratio <= 0.0:
        z_source = self.backbone.predict_latent(z_t, t)
      else:
        z_0 = self.backbone.encode_teacher(input_sequence, t)
        z_hat_0 = self.backbone.predict_latent(z_t, t)
        z_source = mix_ratio * z_0 + (1 - mix_ratio) * z_hat_0
    
    # Decode to logits
    logits = self.backbone.readout_tokens(z_source)
    
    # CE loss on masked positions
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = input_sequence.view(-1)
    per_token_ce = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(B, L)
    
    masked_ce = per_token_ce * mask_indices.float()
    
    if self.weight_by_hazard:
      alpha_t = self.noise.alpha_t(t)
      alpha_prime_t = self.noise.alpha_prime_t(t)
      weight = (alpha_prime_t / (1 - alpha_t).clamp(min=self.sampling_eps)).abs()
      weight = weight.view(B, 1).expand(B, L)
      masked_ce = masked_ce * weight
    
    ce_loss = masked_ce.sum() / num_masked.float()
    
    # Log metrics
    with torch.no_grad():
      preds = logits.argmax(dim=-1)
      correct = (preds == input_sequence).float() * mask_indices.float()
      accuracy = correct.sum() / num_masked.float()
    
    self.log('decoder/ce_loss', ce_loss, on_step=True, on_epoch=False, sync_dist=True)
    self.log('decoder/accuracy', accuracy, on_step=True, on_epoch=False, sync_dist=True)
    self.log('decoder/mix_ratio', mix_ratio, on_step=True, on_epoch=False, sync_dist=True)
    
    num_tokens = valid_tokens.sum()
    return Loss(loss=ce_loss, nlls=ce_loss * num_tokens, num_tokens=num_tokens)

  def _get_mix_ratio(self):
    """Get teacher/predicted latent mix ratio for Stage 2."""
    if self.latent_source == "predicted":
      return 0.0
    elif self.latent_source == "teacher":
      return 1.0
    else:  # mixed
      if self.mixed_schedule == "constant":
        return 0.5
      return max(0.0, 1.0 - self.global_step / max(1, self.mixed_warmup_steps))

  def _vicreg_loss(self, z, mask):
    """VICReg regularization: variance + covariance penalties."""
    B, L, D = z.shape
    device = z.device
    
    z_flat = z[mask] if self.mask_only else z.view(-1, D)
    if z_flat.numel() == 0 or z_flat.shape[0] < 2:
      return torch.tensor(0.0, device=device)
    
    N = z_flat.shape[0]
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    var = z_centered.var(dim=0, unbiased=False)
    var_loss = F.relu(1.0 - var.sqrt()).mean()
    
    safe_std = var.sqrt().clamp(min=1e-3)
    z_std = z_centered / safe_std
    cov = (z_std.T @ z_std) / N
    off_diag = cov.masked_select(~torch.eye(D, device=device, dtype=torch.bool))
    cov_loss = (off_diag ** 2).mean()
    
    reg_loss = self.lambda_var * var_loss + self.lambda_cov * cov_loss
    return reg_loss if torch.isfinite(reg_loss) else torch.tensor(0.0, device=device, dtype=z.dtype)

  def _barlow_loss(self, z, mask):
    """Barlow Twins regularization."""
    B, L, D = z.shape
    device = z.device
    
    z_flat = z[mask] if self.mask_only else z.view(-1, D)
    if z_flat.numel() == 0 or z_flat.shape[0] < 2:
      return torch.tensor(0.0, device=device)
    
    N = z_flat.shape[0]
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    safe_std = z_centered.std(dim=0, keepdim=True).clamp(min=1e-3)
    z_std = z_centered / safe_std
    
    c = (z_std.T @ z_std) / N
    identity = torch.eye(D, device=device)
    on_diag_loss = ((c - identity).diag() ** 2).mean()
    off_diag_mask = ~torch.eye(D, device=device, dtype=torch.bool)
    off_diag_loss = (c[off_diag_mask] ** 2).mean()
    
    reg_loss = self.lambda_var * on_diag_loss + self.lambda_cov * off_diag_loss
    return reg_loss if torch.isfinite(reg_loss) else torch.tensor(0.0, device=device, dtype=z.dtype)

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def nll(self, input_tokens, output_tokens, current_accumulation_step=None, train_mode=False):
    """Not used - _loss is overridden directly."""
    raise NotImplementedError("LatentJEPATrainer overrides _loss directly")

