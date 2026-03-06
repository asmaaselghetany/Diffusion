"""Latent JEPA training algorithm for discrete diffusion.

Supports Stage 1 (latent JEPA training with MSE + VICReg) and 
Stage 2 (decoder-only training with CE loss).
"""

import math

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
    
    # # Noise warmup config
    # self.noise_warmup_start = int(getattr(config.algo, 'noise_warmup_start_step', 0))
    # self.noise_warmup_end = int(getattr(config.algo, 'noise_warmup_end_step', 0))
    # warmup_range = getattr(config.algo, 'noise_warmup_range', [0.15, 0.3])
    # self.noise_warmup_range = [float(warmup_range[0]), float(warmup_range[1])]
    
    # Stage 2 specific
    self.latent_source = getattr(config.algo, 'latent_source', 'predicted').lower()
    self.mixed_schedule = getattr(config.algo, 'mixed_schedule', 'linear').lower()
    self.mixed_warmup_steps = int(getattr(config.algo, 'mixed_warmup_steps', 10000))
    self.weight_by_hazard = getattr(config.algo, 'weight_by_hazard', True)
    
    # Stage 2 mixing study parameters (teacher latent scaffold with rapid handover)
    self.teacher_mix_warm_frac = float(getattr(config.algo, 'teacher_mix_warm_frac', 0.07))
    self.teacher_mix_type = getattr(config.algo, 'teacher_mix_type', 'bernoulli').lower()
    self.teacher_corruption_lambda = float(getattr(config.algo, 'teacher_corruption_lambda', 0.7))
    self.teacher_error_ema_decay = float(getattr(config.algo, 'teacher_error_ema_decay', 0.99))
    
    # Robust teacher corruption parameters
    self.noise_warmup_steps = int(getattr(config.algo, 'noise_warmup_steps', 5000))
    self.noise_min_samples = float(getattr(config.algo, 'noise_min_samples', 1000.0))
    self.noise_max_ratio = float(getattr(config.algo, 'noise_max_ratio', 2.0))
    self.noise_time_scaling = getattr(config.algo, 'noise_time_scaling', True)
    
    # Target normalization layer for I-JEPA best practice
    if self.normalize_targets:
      self.target_norm = nn.LayerNorm(self.backbone.latent_dim)
    
    # Register buffers for Welford statistics (robust error estimation)
    # Will be initialized on first forward pass when we know the device
    self.register_buffer('error_mean', None)
    self.register_buffer('error_m2', None)  # Sum of squared deviations
    self.register_buffer('error_eff_count', None)  # Effective sample count
    self.register_buffer('error_sigma', None)
    self._error_stats_initialized = False
    
    # Freeze encoder/predictor for Stage 2
    if self.stage == 2:
      self._freeze_encoder_predictor()

  def _freeze_encoder_predictor(self):
    """Freeze student encoder and predictor for Stage 2 training."""
    for param in self.backbone.student_encoder.parameters():
      param.requires_grad = False
    for param in self.backbone.predictor.parameters():
      param.requires_grad = False

  # Sampler compatibility shims: latent_jepa sampler expects these methods
  # on the trainer object (not only on self.backbone).
  def encode_student(self, input_ids, t=None, attention_mask=None):
    return self.backbone.encode_student(input_ids, t, attention_mask=attention_mask)

  def predict_latent(self, z_t, t=None):
    return self.backbone.predict_latent(z_t, t)

  def readout_tokens(self, z):
    return self.backbone.readout_tokens(z)

  def _get_parameters(self):
    """Return trainable parameters based on stage."""
    if self.stage == 2:
      return self.backbone.decoder_parameters()
    return self.backbone.optimizable_parameters()

  def _prepare_ema(self):
    """Disable base class EMA in Stage 2.
    
    Stage 2 only trains the readout decoder and doesn't need EMA:
    - The JEPA teacher encoder is frozen (no updates required)
    - The decoder is small; EMA smoothing provides marginal benefit
    - Loading Stage 1's EMA state would cause parameter mismatch errors
    """
    if self.stage == 2:
      self.ema = None
    else:
      super()._prepare_ema()

  def optimizer_step(self, *args, **kwargs):
    """Update optimizer and EMA teacher."""
    super().optimizer_step(*args, **kwargs)
    if self.stage == 1:  # Only update EMA during Stage 1
      self.backbone.update_ema(self.global_step)

  def _sample_timesteps(self, batch_size, device):
    """Sample timesteps with optional noise warmup."""
    # if (self.noise_warmup_end > 0 and 
    #     self.noise_warmup_start <= self.global_step < self.noise_warmup_end):
    #   t_min, t_max = self.noise_warmup_range
    #   t0 = torch.rand((), device=device)
    #   t = (t0 + torch.arange(batch_size, device=device, dtype=torch.float32) / max(batch_size, 1)) % 1.0
    #   t = t_min + t * (t_max - t_min)
    #   return t.clamp(max(t_min, self.sampling_eps), min(t_max, 1.0 - self.sampling_eps))
    # Standard antithetic sampling
    t0 = torch.rand((), device=device)
    t = (t0 + torch.arange(batch_size, device=device, dtype=torch.float32) / max(batch_size, 1)) % 1.0
    return t.clamp(self.sampling_eps, 1.0 - self.sampling_eps)

  def _teacher_timestep(self, t: torch.Tensor | None) -> torch.Tensor | None:
    """Return the teacher conditioning timestep for clean targets.

    Teacher targets are computed from clean inputs, so we anchor time conditioning
    at t=0 instead of reusing the noisy student timestep.
    """
    if t is None:
      return None
    return torch.zeros_like(t)

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
    z_0_raw = self.backbone.encode_teacher(input_sequence, self._teacher_timestep(t))
    
    # Predict clean latents (raw output for VICReg)
    z_hat_0_raw = self.backbone.predict_latent(z_t, t)
    
    # Normalize BOTH teacher and predictor for matching loss (I-JEPA best practice)
    # VICReg uses raw predictor output to preserve variance/covariance signal
    if self.normalize_targets:
      self.target_norm = self.target_norm.to(z_0_raw.device)
      z_0_norm = self.target_norm(z_0_raw)
      z_hat_0_norm = self.target_norm(z_hat_0_raw)
    else:
      z_0_norm = z_0_raw
      z_hat_0_norm = z_hat_0_raw
    
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
    
    # Prediction loss on normalized latents (both sides normalized)
    if self.loss_norm == "l2":
      per_token_loss = ((z_hat_0_norm - z_0_norm) ** 2).mean(dim=-1)
    else:  # cosine
      z_hat_l2 = F.normalize(z_hat_0_norm, p=2, dim=-1)
      z_0_l2 = F.normalize(z_0_norm, p=2, dim=-1)
      per_token_loss = 1 - (z_hat_l2 * z_0_l2).mean(dim=-1)
    
    if self.mask_only:
      weighted_loss = per_token_loss * mask_indices.float() * weight
    else:
      weighted_loss = per_token_loss * weight
    
    pred_loss = weighted_loss.sum() / num_masked.float()
    
    # Regularization loss on RAW predictor output (not normalized)
    # This preserves the variance/covariance signal that VICReg needs
    if self.redundancy == "vicreg":
      reg_loss = self._vicreg_loss(z_hat_0_raw, mask_indices)
    elif self.redundancy == "barlow":
      reg_loss = self._barlow_loss(z_hat_0_raw, mask_indices)
    else:
      reg_loss = torch.tensor(0.0, device=device)
    
    total_loss = pred_loss + reg_loss
    
    # Log metrics with collapse detection (use normalized for cosine sim, raw for collapse detection)
    with torch.no_grad():
      z_hat_l2_log = F.normalize(z_hat_0_norm, p=2, dim=-1)
      z_0_l2_log = F.normalize(z_0_norm, p=2, dim=-1)
      cosine_sim = (z_hat_l2_log * z_0_l2_log).sum(dim=-1)
      if self.mask_only:
        cosine_sim = (cosine_sim * mask_indices.float()).sum() / num_masked.float()
      else:
        cosine_sim = cosine_sim.mean()
      
      # Collapse detection metrics (use raw outputs to detect true collapse)
      z_hat_masked = z_hat_0_raw[mask_indices] if self.mask_only else z_hat_0_raw.view(-1, z_hat_0_raw.size(-1))
      z_0_masked = z_0_raw[mask_indices] if self.mask_only else z_0_raw.view(-1, z_0_raw.size(-1))
      
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
    """Stage 2: Decoder CE loss on masked positions with teacher mixing study.
    
    Implements front-loaded teacher latent mixing:
    - Bernoulli sampling: at each batch, either use teacher z_0 OR predicted z_hat_0
    - Cosine annealing: teacher probability decays from 1 to 0 over warm window
    - Teacher corruption: add noise to teacher latents to prevent brittle shortcuts
    """
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
      z_0_raw = self.backbone.encode_teacher(input_sequence, self._teacher_timestep(t))
      z_hat_0_raw = self.backbone.predict_latent(z_t, t)
      
      # Normalize BOTH teacher and predictor latents for decoder input
      # This ensures decoder sees consistent distributions regardless of source
      if self.normalize_targets:
        self.target_norm = self.target_norm.to(z_0_raw.device)
        z_0 = self.target_norm(z_0_raw)
        z_hat_0 = self.target_norm(z_hat_0_raw)
      else:
        z_0 = z_0_raw
        z_hat_0 = z_hat_0_raw
      
      # Update error sigma EMA for teacher corruption (use normalized latents)
      self._update_error_sigma(z_0, z_hat_0)
      
      # Determine whether to use teacher or predicted latents
      beta = self._get_teacher_beta()
      
      if self.teacher_mix_type == 'bernoulli':
        # Bernoulli sampling: bimodal distribution (teacher vs predicted)
        u = torch.rand((), device=device)
        use_teacher = (u < beta)
        
        if use_teacher:
          # Apply teacher corruption (hardening trick)
          z_source = self._corrupt_teacher_latent(z_0, t=t)
        else:
          z_source = z_hat_0
      else:
        # Legacy blend mode: linear interpolation
        if beta >= 1.0:
          z_source = z_0
        elif beta <= 0.0:
          z_source = z_hat_0
        else:
          z_source = beta * z_0 + (1 - beta) * z_hat_0
        use_teacher = (beta > 0.5)
    
    # Enable gradients for readout backward pass (required for gradient checkpointing + DDP)
    z_source = z_source.requires_grad_(True)
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
      
      # Log error sigma statistics
      error_sigma_mean = self.error_sigma.mean() if self.error_sigma is not None else 0.0
    
    self.log('decoder/ce_loss', ce_loss, on_step=True, on_epoch=False, sync_dist=True)
    self.log('decoder/accuracy', accuracy, on_step=True, on_epoch=False, sync_dist=True)
    self.log('decoder/teacher_beta', beta, on_step=True, on_epoch=False, sync_dist=True)
    self.log('decoder/use_teacher', float(use_teacher), on_step=True, on_epoch=False, sync_dist=True)
    self.log('decoder/error_sigma_mean', error_sigma_mean, on_step=True, on_epoch=False, sync_dist=True)
    
    num_tokens = valid_tokens.sum()
    return Loss(loss=ce_loss, nlls=ce_loss * num_tokens, num_tokens=num_tokens)

  def _update_error_sigma(self, z_0, z_hat_0):
    """Update error statistics using batched Welford with exponential forgetting.
    
    Accumulates statistics across batches for stable estimation while
    allowing adaptation as the predictor improves during training.
    """
    error = z_hat_0 - z_0
    error_flat = error.view(-1, error.size(-1))
    N, D = error_flat.shape
    
    # Initialize buffers on first call
    if not self._error_stats_initialized:
      self.error_mean = torch.zeros(D, device=error.device, dtype=error.dtype)
      self.error_m2 = torch.zeros(D, device=error.device, dtype=error.dtype)
      self.error_eff_count = torch.tensor(0.0, device=error.device)
      self.error_sigma = torch.ones(D, device=error.device, dtype=error.dtype)
      self._error_stats_initialized = True
    
    # Batch statistics
    batch_mean = error_flat.mean(dim=0)
    batch_var = error_flat.var(dim=0, unbiased=False)
    
    # Parallel Welford update with exponential forgetting
    decay = self.teacher_error_ema_decay
    n_a = self.error_eff_count * decay  # Decayed history count
    n_b = float(N)                       # New batch count
    n_total = n_a + n_b
    
    delta = batch_mean - self.error_mean
    
    # Update mean
    self.error_mean = self.error_mean + delta * (n_b / n_total)
    
    # Update M2 (parallel variance combination formula)
    batch_m2 = batch_var * n_b
    self.error_m2 = self.error_m2 * decay + batch_m2 + (delta ** 2) * n_a * n_b / n_total
    self.error_eff_count = torch.tensor(n_total, device=error.device)
    
    # Compute sigma only when we have sufficient samples
    if self.error_eff_count > self.noise_min_samples:
      self.error_sigma = (self.error_m2 / self.error_eff_count).sqrt().clamp(min=1e-6)

  def _corrupt_teacher_latent(self, z_0, t=None):
    """Add robust corruption noise to teacher latent.
    
    Features:
    - Warmup: gradually increase noise over early training
    - Time-dependent scaling: more noise at high t (harder predictions)
    - Magnitude clipping: prevent extreme perturbations
    - Aligned with teacher beta for smooth transitions
    """
    if self.error_sigma is None or self.teacher_corruption_lambda <= 0:
      return z_0
    
    # Skip noise if insufficient statistics accumulated
    if self.error_eff_count is not None and self.error_eff_count < self.noise_min_samples:
      return z_0
    
    # 1. Warmup factor: ramp from 0 to 1 over warmup steps
    warmup = min(1.0, self.global_step / max(1, self.noise_warmup_steps))
    
    # 2. Base noise std from accumulated statistics
    base_std = self.teacher_corruption_lambda * self.error_sigma  # (D,)
    
    # 3. Time-dependent scaling (optional)
    if self.noise_time_scaling and t is not None:
      # More noise at high t (more masking = harder prediction)
      # Scale by sqrt(t) to match diffusion noise scaling intuition
      t_scale = t.view(-1, 1, 1).sqrt().clamp(min=0.1)  # (B, 1, 1)
      noise_std = warmup * base_std.view(1, 1, -1) * t_scale
    else:
      noise_std = warmup * base_std.view(1, 1, -1)
    
    # 4. Generate noise
    epsilon = torch.randn_like(z_0) * noise_std
    
    # 5. Clip extreme noise to prevent outlier perturbations
    z_std = z_0.std(dim=-1, keepdim=True).clamp(min=1e-6)
    max_eps = self.noise_max_ratio * z_std
    epsilon = epsilon.clamp(-max_eps, max_eps)
    
    return z_0 + epsilon

  def _get_mix_ratio(self):
    """Get teacher/predicted latent mix ratio for Stage 2 (legacy linear schedule)."""
    if self.latent_source == "predicted":
      return 0.0
    elif self.latent_source == "teacher":
      return 1.0
    else:  # mixed
      if self.mixed_schedule == "constant":
        return 0.5
      return max(0.0, 1.0 - self.global_step / max(1, self.mixed_warmup_steps))

  def _get_total_steps(self):
    """Get total training steps for progress calculation."""
    # Try to get from trainer (PyTorch Lightning)
    if hasattr(self, 'trainer') and self.trainer is not None:
      if hasattr(self.trainer, 'max_steps') and self.trainer.max_steps > 0:
        return self.trainer.max_steps
      # Fallback to estimated stepping batches
      try:
        return self.trainer.estimated_stepping_batches
      except Exception:
        pass
    # Fallback to mixed_warmup_steps as denominator
    return max(1, self.mixed_warmup_steps)

  def _get_teacher_beta(self):
    """Compute teacher mixing probability beta(s) with cosine annealing.
    
    Implements fast cosine anneal schedule:
      beta(s) = 0.5 * (1 + cos(pi * s / s_w))  for s <= s_w
      beta(s) = 0                               for s > s_w
    
    where s = global_step / total_steps is training progress,
    and s_w = teacher_mix_warm_frac is the warm window fraction.
    
    Returns:
      float: Teacher mixing probability in [0, 1]
    """
    if self.latent_source == "predicted":
      return 0.0
    elif self.latent_source == "teacher":
      return 1.0
    
    # Compute progress s ∈ [0, 1]
    total_steps = self._get_total_steps()
    s = self.global_step / max(1, total_steps)
    s_w = self.teacher_mix_warm_frac
    
    if s > s_w:
      # After warm window: 100% predicted latents
      return 0.0
    else:
      # During warm window: cosine annealing from 1 to 0
      return 0.5 * (1.0 + math.cos(math.pi * s / max(s_w, 1e-6)))

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

  def validation_step(self, batch, batch_idx):
    """Override validation step for Stage 2 to log dual CE metrics.
    
    Logs:
      - val/ce_teacher: CE with decoder input = z_0 (upper bound)
      - val/ce_predicted: CE with decoder input = z_hat_0 (real objective)
    """
    if self.stage != 2:
      # Stage 1: use base class validation
      return super().validation_step(batch, batch_idx)
    
    input_sequence = batch['input_ids']
    valid_tokens = batch['attention_mask']
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
    
    with torch.no_grad():
      z_t = self.backbone.encode_student(x_t, t)
      z_0_raw = self.backbone.encode_teacher(input_sequence, self._teacher_timestep(t))
      z_hat_0_raw = self.backbone.predict_latent(z_t, t)
      
      # Normalize BOTH teacher and predictor latents for decoder input
      # This ensures decoder sees consistent distributions regardless of source
      if self.normalize_targets:
        self.target_norm = self.target_norm.to(z_0_raw.device)
        z_0 = self.target_norm(z_0_raw)
        z_hat_0 = self.target_norm(z_hat_0_raw)
      else:
        z_0 = z_0_raw
        z_hat_0 = z_hat_0_raw
      
      # Compute CE with teacher latents (upper bound)
      logits_teacher = self.backbone.readout_tokens(z_0)
      flat_logits_teacher = logits_teacher.view(-1, logits_teacher.size(-1))
      flat_targets = input_sequence.view(-1)
      per_token_ce_teacher = F.cross_entropy(flat_logits_teacher, flat_targets, reduction='none').view(B, L)
      masked_ce_teacher = (per_token_ce_teacher * mask_indices.float()).sum() / num_masked.float()
      
      # Compute CE with predicted latents (real objective)
      logits_pred = self.backbone.readout_tokens(z_hat_0)
      flat_logits_pred = logits_pred.view(-1, logits_pred.size(-1))
      per_token_ce_pred = F.cross_entropy(flat_logits_pred, flat_targets, reduction='none').view(B, L)
      masked_ce_pred = (per_token_ce_pred * mask_indices.float()).sum() / num_masked.float()
      
      # Compute accuracy with predicted latents
      preds = logits_pred.argmax(dim=-1)
      correct = (preds == input_sequence).float() * mask_indices.float()
      accuracy = correct.sum() / num_masked.float()
    
    # Log dual CE metrics
    self.log('val/ce_teacher', masked_ce_teacher, on_step=False, on_epoch=True, sync_dist=True)
    self.log('val/ce_predicted', masked_ce_pred, on_step=False, on_epoch=True, sync_dist=True)
    self.log('val/accuracy', accuracy, on_step=False, on_epoch=True, sync_dist=True)
    
    # Update validation metrics (use predicted CE as main metric)
    num_tokens = valid_tokens.sum()
    self.metrics.update_valid(masked_ce_pred * num_tokens, num_tokens)
    
    return masked_ce_pred
