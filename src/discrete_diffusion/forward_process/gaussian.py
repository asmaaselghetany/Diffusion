"""Gaussian forward process for continuous embedding diffusion.

Implements the standard DDPM-style Gaussian noising:
  z_t = sqrt(alpha_t) * z_0 + sqrt(1 - alpha_t) * epsilon

where epsilon ~ N(0, I) and alpha_t is the signal retention schedule.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn

from ..noise_schedules.base import NoiseSchedule


class GaussianForwardProcess(nn.Module):
    """Continuous Gaussian forward process for embedding diffusion.
    
    This process adds Gaussian noise to continuous embeddings according
    to a noise schedule, producing noisy embeddings z_t from clean z_0.
    
    The forward process follows the DDPM formulation:
        z_t = sqrt(alpha_t) * z_0 + sqrt(1 - alpha_t) * epsilon
    
    where:
        - z_0: Clean embeddings [B, L, D] or [B, D]
        - epsilon: Standard Gaussian noise
        - alpha_t: Signal retention coefficient from schedule
        - z_t: Noisy embeddings
    
    Attributes:
        schedule: Noise schedule providing alpha_t values
        clip_noise: Whether to clip extreme noise values (numerical stability)
        noise_clip_value: Maximum absolute value for noise clipping
    """
    
    def __init__(
        self,
        schedule: NoiseSchedule,
        clip_noise: bool = False,
        noise_clip_value: float = 6.0,
        per_token_noise: bool = True,
    ):
        """Initialize Gaussian forward process.
        
        Args:
            schedule: Noise schedule providing alpha_t(t) and alpha_prime_t(t)
            clip_noise: Whether to clip extreme noise values for stability
            noise_clip_value: Max absolute value for noise if clipping enabled
            per_token_noise: If True, sample independent noise per token position
        """
        super().__init__()
        self.schedule = schedule
        self.clip_noise = clip_noise
        self.noise_clip_value = noise_clip_value
        self.per_token_noise = per_token_noise
    
    def _get_alpha_t(self, t: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Get alpha_t broadcasted to target shape.
        
        Args:
            t: Timesteps [B] or scalar
            target_shape: Target tensor shape for broadcasting
            
        Returns:
            alpha_t tensor broadcastable to target_shape
        """
        alpha_t = self.schedule.alpha_t(t)
        
        # Ensure t is at least 1D
        if alpha_t.ndim == 0:
            alpha_t = alpha_t.unsqueeze(0)
        
        # Broadcast to match input dimensions
        # For [B, L, D] input, we need [B, 1, 1]
        # For [B, D] input, we need [B, 1]
        n_expand = len(target_shape) - 1
        for _ in range(n_expand):
            alpha_t = alpha_t.unsqueeze(-1)
        
        return alpha_t
    
    def forward(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        return_noise: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply Gaussian forward process to clean embeddings.
        
        Args:
            z_0: Clean embeddings [B, L, D] or [B, D]
            t: Timesteps [B] with values in [0, 1]
            return_noise: Whether to return the sampled noise
            
        Returns:
            Tuple of:
                - z_t: Noisy embeddings, same shape as z_0
                - epsilon: Sampled noise (if return_noise=True), else None
        """
        # Sample Gaussian noise
        epsilon = torch.randn_like(z_0)
        
        # Optionally clip extreme noise values
        if self.clip_noise:
            epsilon = epsilon.clamp(-self.noise_clip_value, self.noise_clip_value)
        
        # Get schedule coefficients broadcasted to z_0 shape
        alpha_t = self._get_alpha_t(t, z_0.shape)
        
        # Compute noisy embeddings: z_t = sqrt(alpha_t) * z_0 + sqrt(1 - alpha_t) * epsilon
        sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-20))
        sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-20))
        
        z_t = sqrt_alpha_t * z_0 + sqrt_one_minus_alpha_t * epsilon
        
        if return_noise:
            return z_t, epsilon
        return z_t, None
    
    def q_sample(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample z_t ~ q(z_t | z_0) with optional pre-sampled noise.
        
        This is a convenience method for cases where noise is pre-computed
        (e.g., for loss computation with same noise across multiple calls).
        
        Args:
            z_0: Clean embeddings [B, L, D] or [B, D]
            t: Timesteps [B] with values in [0, 1]
            noise: Optional pre-sampled noise, same shape as z_0
            
        Returns:
            z_t: Noisy embeddings
        """
        if noise is None:
            noise = torch.randn_like(z_0)
            if self.clip_noise:
                noise = noise.clamp(-self.noise_clip_value, self.noise_clip_value)
        
        alpha_t = self._get_alpha_t(t, z_0.shape)
        sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-20))
        sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-20))
        
        return sqrt_alpha_t * z_0 + sqrt_one_minus_alpha_t * noise
    
    def q_posterior_mean_variance(
        self,
        z_0: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior q(z_{t-1} | z_t, z_0) mean and variance.
        
        For DDPM-style reverse process, computes:
            mu = (sqrt(alpha_{t-1}) * beta_t * z_0 + sqrt(alpha_t) * (1-alpha_{t-1}) * z_t)
                 / (1 - alpha_t)
            var = beta_t * (1 - alpha_{t-1}) / (1 - alpha_t)
        
        where beta_t = 1 - alpha_t / alpha_{t-1}
        
        Args:
            z_0: Predicted clean embeddings [B, L, D] or [B, D]
            z_t: Current noisy embeddings, same shape
            t: Current timesteps [B]
            t_prev: Previous timesteps [B] (t_prev < t)
            
        Returns:
            Tuple of (posterior_mean, posterior_variance)
        """
        alpha_t = self._get_alpha_t(t, z_0.shape)
        alpha_t_prev = self._get_alpha_t(t_prev, z_0.shape)
        
        # beta_t = 1 - alpha_t / alpha_{t-1}
        beta_t = 1.0 - alpha_t / alpha_t_prev.clamp(min=1e-20)
        
        # Posterior mean
        coef_z0 = torch.sqrt(alpha_t_prev.clamp(min=1e-20)) * beta_t / (1.0 - alpha_t).clamp(min=1e-20)
        coef_zt = torch.sqrt(alpha_t.clamp(min=1e-20)) * (1.0 - alpha_t_prev) / (1.0 - alpha_t).clamp(min=1e-20)
        posterior_mean = coef_z0 * z_0 + coef_zt * z_t
        
        # Posterior variance
        posterior_variance = beta_t * (1.0 - alpha_t_prev) / (1.0 - alpha_t).clamp(min=1e-20)
        
        return posterior_mean, posterior_variance
    
    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Compute signal-to-noise ratio at time t.
        
        SNR(t) = alpha_t / (1 - alpha_t)
        
        Args:
            t: Timesteps [B] or scalar
            
        Returns:
            SNR values, same shape as t
        """
        alpha_t = self.schedule.alpha_t(t)
        return alpha_t / (1.0 - alpha_t).clamp(min=1e-20)
    
    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Compute log signal-to-noise ratio at time t.
        
        log_SNR(t) = log(alpha_t) - log(1 - alpha_t)
        
        Args:
            t: Timesteps [B] or scalar
            
        Returns:
            log SNR values, same shape as t
        """
        alpha_t = self.schedule.alpha_t(t)
        return torch.log(alpha_t.clamp(min=1e-20)) - torch.log((1.0 - alpha_t).clamp(min=1e-20))


__all__ = ["GaussianForwardProcess"]
