"""Continuous embedding sampler for Gaussian diffusion.

Implements DDPM/DDIM/ancestral sampling with centralized VP math utilities.
"""

from __future__ import annotations

from typing import Any, Optional, Literal, Tuple

import torch
import torch.nn.functional as F

from .base import Sampler
from ..continuous.math import (
    build_vp_schedule_from_alphabar,
    scale_alphabar_noise,
    eps_from_x0,
    q_posterior,
)


class ContinuousEmbeddingSampler(Sampler):
    """Sampler for continuous embedding diffusion models.
    
    Supports multiple sampling strategies:
        - DDPM: Stochastic reverse process with full noise injection
        - DDIM: Deterministic reverse process (faster, same quality)
        - Ancestral: Similar to DDPM but with different noise schedule
    
    The sampler operates in continuous embedding space and decodes to
    discrete tokens at the end of sampling.
    """
    
    def __init__(
        self,
        config,
        forward_process=None,
        sampling_method: Literal[
            "ddpm",
            "ddim",
            "ancestral",
            "rectified_one_step",
            "rectified_few_step",
            "flowmap_one_step",
            "flowmap_few_step",
        ] = "ddpm",
        eta: float = 1.0,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        clip_denoised: bool = False,
        clip_range: Tuple[float, float] = (-10.0, 10.0),
        rectified_init: Literal["random_normal", "fixed_common"] = "random_normal",
        rectified_common_token_id: Optional[int] = None,
        top_p: float = 0.0,
    ):
        """Initialize the continuous embedding sampler.
        
        Args:
            config: Hydra config
            forward_process: Forward process (GaussianForwardProcess)
            sampling_method: 'ddpm', 'ddim', 'ancestral', 'rectified_one_step', or 'rectified_few_step'
            eta: DDIM eta parameter (0=deterministic, 1=DDPM-like)
            temperature: Softmax temperature for token selection
            guidance_scale: Classifier-free guidance scale (>1 for stronger guidance)
            clip_denoised: Whether to clip denoised predictions
            clip_range: Range for clipping if enabled
            rectified_init: Initialization for rectified samplers
            rectified_common_token_id: Token id used when rectified_init='fixed_common'
            top_p: Nucleus sampling threshold in (0, 1]. 0 = greedy argmax.
        """
        self.config = config
        self.forward_process = forward_process
        self.sampling_method = sampling_method
        self.eta = eta
        self.temperature = temperature
        self.guidance_scale = guidance_scale
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range
        self.rectified_init = rectified_init
        self.rectified_common_token_id = rectified_common_token_id
        self.top_p = top_p

    def _init_rectified_latents(
        self,
        model: Any,
        *,
        num_samples: int,
        seq_len: int,
        embed_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.rectified_init == "random_normal":
            return torch.randn(num_samples, seq_len, embed_dim, device=device)
        if self.rectified_init != "fixed_common":
            raise ValueError(f"Unknown rectified_init='{self.rectified_init}'")

        token_id = 0 if self.rectified_common_token_id is None else int(self.rectified_common_token_id)
        ids = torch.full((num_samples, seq_len), token_id, device=device, dtype=torch.long)
        attn = torch.ones_like(ids)
        if hasattr(model, "embedding_provider") and getattr(model, "embedding_provider_name", "legacy_contextual") in {"lookup", "tied"}:
            return model.embedding_provider.embed(ids)
        if hasattr(model, "_embed_inputs"):
            return model._embed_inputs(ids, attn)
        return torch.randn(num_samples, seq_len, embed_dim, device=device)

    @staticmethod
    def _is_flow_matching_model(model: Any) -> bool:
        objective_name = str(getattr(model, "objective_name", "")).lower()
        return objective_name == "flow_matching"

    def _predict_rectified_direction(
        self,
        model: Any,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the velocity-like direction used by rectified samplers.

        - Flow matching trains instantaneous velocity `v(z,t)`.
        - MeanFlow/iMF trains average velocity `u(z,r,t)`.
        """
        if self._is_flow_matching_model(model):
            if hasattr(model, "predict_velocity"):
                return model.predict_velocity(z_t, t)
            return model(z_t, t)

        if r is None:
            r = torch.zeros_like(t)
        if hasattr(model, "predict_u"):
            return model.predict_u(z_t, r, t)
        return model(z_t, t, r=r)

    @staticmethod
    def _predict_flowmap_position(
        model: Any,
        z_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Direct position prediction: F_θ(z_t, t, s) → z_s.

        Uses the model's default forward (parameterization='x0' with residual)
        which returns net_θ(z_t, t, r=s) + z_t.
        """
        return model(z_t, t, r=s)

    def _decode_latents(
        self,
        model: Any,
        z: torch.Tensor,
        return_embeddings: bool,
    ) -> torch.Tensor:
        """Decode latent embeddings to tokens (or return raw embeddings)."""
        if return_embeddings:
            return z
        if hasattr(model, "embedding_provider") and getattr(model, "embedding_provider_name", "legacy_contextual") in {"lookup", "tied"}:
            logits = model.embedding_provider.logits(z)
        else:
            logits = model.decoder(z)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        if self.top_p > 0.0:
            return self._sample_nucleus(logits, self.top_p)
        return logits.argmax(dim=-1)

    @staticmethod
    def _sample_nucleus(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus (top-p) sampling over the last dimension of logits.

        Args:
            logits: [B, L, V] raw (temperature-scaled) logits.
            top_p: Cumulative probability threshold in (0, 1].

        Returns:
            Sampled token ids [B, L].
        """
        B, L, V = logits.shape
        flat = logits.view(-1, V)
        probs = F.softmax(flat, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)

        # Zero out tokens beyond the nucleus
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        sampled_sorted_pos = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        tokens = sorted_idx.gather(dim=-1, index=sampled_sorted_pos.unsqueeze(-1)).squeeze(-1)
        return tokens.view(B, L)

    def generate(
        self,
        model: Any,
        *,
        num_samples: int,
        num_steps: int,
        eps: float = 1e-5,
        inject_bos: bool = False,
        seq_len: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """Generate samples using reverse diffusion.
        
        Args:
            model: ContinuousEmbeddingDiffusion model
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps
            eps: Small value to avoid numerical issues
            inject_bos: Whether to inject BOS token (not applicable for continuous)
            seq_len: Sequence length (defaults to config)
            return_embeddings: If True, return embeddings instead of tokens
            
        Returns:
            Generated token IDs [num_samples, seq_len] or embeddings if return_embeddings
        """
        # Get model parameters
        device = next(model.denoiser.parameters()).device
        embed_dim = model.embed_dim
        
        if seq_len is None:
            seq_len = getattr(self.config.model, 'length', 256)
        
        if self.sampling_method == "rectified_one_step":
            z_1 = self._init_rectified_latents(
                model,
                num_samples=num_samples,
                seq_len=seq_len,
                embed_dim=embed_dim,
                device=device,
            )
            t = torch.ones(num_samples, device=device)
            r = torch.zeros(num_samples, device=device)
            direction = self._predict_rectified_direction(model, z_1, t, r=r)
            z_t = z_1 - direction
            return self._decode_latents(model, z_t, return_embeddings)

        if self.sampling_method == "rectified_few_step":
            z_t = self._init_rectified_latents(
                model,
                num_samples=num_samples,
                seq_len=seq_len,
                embed_dim=embed_dim,
                device=device,
            )
            times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
            for i in range(num_steps):
                t_k = times[i].expand(num_samples)
                t_km1 = times[i + 1].expand(num_samples)
                direction = self._predict_rectified_direction(model, z_t, t_k, r=t_km1)
                dt = (t_k - t_km1).view(-1, 1, 1)
                z_t = z_t - dt * direction
            return self._decode_latents(model, z_t, return_embeddings)

        if self.sampling_method == "flowmap_one_step":
            z_1 = self._init_rectified_latents(
                model,
                num_samples=num_samples,
                seq_len=seq_len,
                embed_dim=embed_dim,
                device=device,
            )
            t = torch.ones(num_samples, device=device)
            s = torch.zeros(num_samples, device=device)
            z_t = self._predict_flowmap_position(model, z_1, t, s)
            return self._decode_latents(model, z_t, return_embeddings)

        if self.sampling_method == "flowmap_few_step":
            z_t = self._init_rectified_latents(
                model,
                num_samples=num_samples,
                seq_len=seq_len,
                embed_dim=embed_dim,
                device=device,
            )
            times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
            for i in range(num_steps):
                t_k = times[i].expand(num_samples)
                s_k = times[i + 1].expand(num_samples)
                z_t = self._predict_flowmap_position(model, z_t, t_k, s_k)
            return self._decode_latents(model, z_t, return_embeddings)

        # VP/DDPM family: initialize from pure Gaussian noise.
        z_t = torch.randn(num_samples, seq_len, embed_dim, device=device)

        # Create reverse timestep schedule (high noise -> low noise).
        timesteps = torch.linspace(1.0 - eps, eps, num_steps + 1, device=device)
        # Build a forward (clean -> noisy) discrete schedule for centralized math.
        timesteps_fwd = torch.flip(timesteps, dims=[0])
        alphabar_fwd = model.noise.alpha_t(timesteps_fwd).float().clamp(min=1e-20, max=1.0)
        noise_scale = float(getattr(model, "noise_scale", 1.0))
        alphabar_fwd = scale_alphabar_noise(alphabar_fwd, noise_scale)
        vp_schedule = build_vp_schedule_from_alphabar(alphabar_fwd)

        # Reverse diffusion loop
        for i in range(num_steps):
            t = timesteps[i].expand(num_samples)
            t_prev = timesteps[i + 1].expand(num_samples)
            # Map reverse index to forward VP timestep index.
            # i=0 (most noisy) => k=num_steps, i=num_steps-1 => k=1.
            k = num_steps - i
            k_prev = max(k - 1, 0)
            k_t = torch.full((num_samples,), k, device=device, dtype=torch.long)
            k_prev_t = torch.full((num_samples,), k_prev, device=device, dtype=torch.long)
            
            # Perform one denoising step
            if self.sampling_method == "ddpm":
                z_t = self._ddpm_step(model, z_t, t, k_t, k_prev_t, vp_schedule)
            elif self.sampling_method == "ddim":
                z_t = self._ddim_step(model, z_t, t, k_t, k_prev_t, vp_schedule)
            elif self.sampling_method == "ancestral":
                z_t = self._ancestral_step(model, z_t, t, k_t, k_prev_t, vp_schedule)
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")
        
        return self._decode_latents(model, z_t, return_embeddings)
    
    def _get_alpha_t(
        self,
        model: Any,
        t: torch.Tensor,
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Get alpha_t broadcasted to target shape."""
        alpha_t = model.noise.alpha_t(t)
        while alpha_t.ndim < len(target_shape):
            alpha_t = alpha_t.unsqueeze(-1)
        return alpha_t
    
    def _predict_z0(
        self,
        model: Any,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict z_0 from z_t using the model."""
        alpha_t = self._get_alpha_t(model, t, z_t.shape)
        z_hat_0 = model.denoiser.predict_x0(z_t, t, alpha_t)
        
        # Optional clipping
        if self.clip_denoised:
            z_hat_0 = z_hat_0.clamp(self.clip_range[0], self.clip_range[1])
        
        return z_hat_0
    
    def _ddpm_step(
        self,
        model: Any,
        z_t: torch.Tensor,
        t_model: torch.Tensor,
        t_idx: torch.Tensor,
        t_prev_idx: torch.Tensor,
        vp_schedule,
    ) -> torch.Tensor:
        """DDPM reverse step.
        
        Computes z_{t-1} ~ p(z_{t-1} | z_t) using the posterior distribution.
        """
        z_hat_0 = self._predict_z0(model, z_t, t_model)
        posterior_mean, posterior_variance, _ = q_posterior(
            x0=z_hat_0,
            x_t=z_t,
            t=t_idx,
            sched=vp_schedule,
        )
        if int(t_prev_idx.max().item()) > 0:
            return posterior_mean + torch.sqrt(posterior_variance.clamp(min=1e-20)) * torch.randn_like(z_t)
        return posterior_mean
    
    def _ddim_step(
        self,
        model: Any,
        z_t: torch.Tensor,
        t_model: torch.Tensor,
        t_idx: torch.Tensor,
        t_prev_idx: torch.Tensor,
        vp_schedule,
    ) -> torch.Tensor:
        """DDIM reverse step.
        
        Deterministic reverse step (when eta=0) or stochastic (when eta>0).
        """
        z_hat_0 = self._predict_z0(model, z_t, t_model)
        epsilon = eps_from_x0(x_t=z_t, x0=z_hat_0, t=t_idx, sched=vp_schedule)

        ab_t = vp_schedule.alphabar.index_select(0, t_idx.view(-1)).to(device=z_t.device, dtype=z_t.dtype)
        ab_prev = vp_schedule.alphabar.index_select(0, t_prev_idx.view(-1)).to(device=z_t.device, dtype=z_t.dtype)
        while ab_t.ndim < z_t.ndim:
            ab_t = ab_t.unsqueeze(-1)
            ab_prev = ab_prev.unsqueeze(-1)

        sigma = self.eta * torch.sqrt(
            ((1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-20))
            * (1.0 - (ab_t / ab_prev.clamp(min=1e-20)))
        )
        pred_direction = torch.sqrt((1.0 - ab_prev - sigma ** 2).clamp(min=0.0)) * epsilon
        z_t_prev = torch.sqrt(ab_prev.clamp(min=1e-20)) * z_hat_0 + pred_direction
        if self.eta > 0 and int(t_prev_idx.max().item()) > 0:
            z_t_prev = z_t_prev + sigma * torch.randn_like(z_t)
        return z_t_prev
    
    def _ancestral_step(
        self,
        model: Any,
        z_t: torch.Tensor,
        t_model: torch.Tensor,
        t_idx: torch.Tensor,
        t_prev_idx: torch.Tensor,
        vp_schedule,
    ) -> torch.Tensor:
        """Ancestral sampling step.
        
        Similar to DDPM but with a different noise formulation.
        """
        z_hat_0 = self._predict_z0(model, z_t, t_model)
        epsilon = eps_from_x0(x_t=z_t, x0=z_hat_0, t=t_idx, sched=vp_schedule)

        ab_t = vp_schedule.alphabar.index_select(0, t_idx.view(-1)).to(device=z_t.device, dtype=z_t.dtype)
        ab_prev = vp_schedule.alphabar.index_select(0, t_prev_idx.view(-1)).to(device=z_t.device, dtype=z_t.dtype)
        while ab_t.ndim < z_t.ndim:
            ab_t = ab_t.unsqueeze(-1)
            ab_prev = ab_prev.unsqueeze(-1)

        z_t_prev = torch.sqrt(ab_prev.clamp(min=1e-20)) * z_hat_0 + torch.sqrt((1.0 - ab_prev).clamp(min=1e-20)) * epsilon
        if int(t_prev_idx.max().item()) > 0:
            noise_scale = torch.sqrt(
                ((1.0 - ab_prev) * (1.0 - (ab_t / ab_prev.clamp(min=1e-20))))
                / (1.0 - ab_t).clamp(min=1e-20)
            )
            z_t_prev = z_t_prev + self.eta * noise_scale * torch.randn_like(z_t)
        return z_t_prev
    
    def generate_with_guidance(
        self,
        model: Any,
        *,
        num_samples: int,
        num_steps: int,
        conditioning_embeddings: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples with classifier-free guidance.
        
        Args:
            model: ContinuousEmbeddingDiffusion model
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps
            conditioning_embeddings: Optional conditioning signal
            eps: Small value to avoid numerical issues
            seq_len: Sequence length
            
        Returns:
            Generated token IDs [num_samples, seq_len]
        """
        # For now, fall back to standard generation
        # CFG requires unconditional training which isn't implemented yet
        return self.generate(
            model,
            num_samples=num_samples,
            num_steps=num_steps,
            eps=eps,
            seq_len=seq_len,
        )
    
    def compute_posterior(self, x: Any, t: Any, dt: Any, p_x0_cache: Optional[Any]) -> Any:
        """Not implemented for continuous sampler."""
        raise NotImplementedError("Use generate() method instead")
    
    def step_analytic(self, x: Any, t: Any, dt: Any) -> Any:
        """Not implemented for continuous sampler."""
        raise NotImplementedError("Use generate() method instead")
    
    def denoise(self, x: Any, t: Any) -> Any:
        """Not implemented for continuous sampler."""
        raise NotImplementedError("Use generate() method instead")


__all__ = ["ContinuousEmbeddingSampler"]
