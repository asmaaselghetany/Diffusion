"""Continuous Embedding Diffusion training algorithm.

Trains a denoiser on continuous embeddings from a frozen pretrained encoder
(e.g., Qwen3-Embedding), using Gaussian forward process and MSE loss.
"""

from __future__ import annotations

import itertools
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import hydra.utils
import omegaconf

from ..evaluations import Metrics
from ..models import create_ema
from ..models.common import set_sdpa_math_mode
from ..models.qwen_embedding_provider import Qwen3EmbeddingProvider
from ..models.continuous_denoiser import ContinuousDenoiser, TransformerDecoder
from ..forward_process.gaussian import GaussianForwardProcess
from ..continuous import (
    DDPMObjective,
    DDPMObjectiveConfig,
    FlowMapObjective,
    FlowMapObjectiveConfig,
    FlowMatchingObjective,
    FlowMatchingObjectiveConfig,
    IMFObjective,
    IMFObjectiveConfig,
    ContextualEmbeddingProviderAdapter,
    build_lookup_or_tied_provider,
    build_vp_schedule_from_alphabar,
    scale_alphabar_noise,
    SpanMasker,
    SpanMaskerConfig,
    run_sampler_smoke,
    probe_model_jvp,
)


@dataclass
class Loss:
    """Container for loss components."""
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    num_tokens: torch.FloatTensor
    mse_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    aux_losses: Optional[dict[str, torch.FloatTensor]] = None


class ContinuousEmbeddingDiffusion(L.LightningModule):
    """Continuous Embedding Diffusion trainer.
    
    This algorithm trains in two explicit stages:
      - Stage 1: train denoiser only in latent space with diffusion MSE.
      - Stage 2: train decoder only with configurable latent source.
    
    Training pipeline:
        Stage 1:
            1. Encode text with frozen encoder -> z_0
            2. Add Gaussian noise via forward process -> z_t
            3. Predict diffusion target from z_t
            4. Compute MSE loss (x0/epsilon/v parameterization)
        Stage 2:
            1. Encode text with frozen encoder -> z_0
            2. Select decoder latents from:
               - teacher z_0
               - denoiser-predicted x_0 from diffused z_t
               - mixed teacher/predicted schedule
            3. Decode selected latents to tokens with CE supervision
    
    Supports:
        - Multiple parameterizations: x0, epsilon, v
        - Multiple loss types: simple (MSE), weighted (ELBO-style)
        - Two-stage training: denoiser pretraining then decoder training
    """
    
    def __init__(self, config, tokenizer):
        """Initialize the continuous embedding diffusion trainer.
        
        Args:
            config: Hydra config with algorithm and model parameters
            tokenizer: Tokenizer for the embedding model
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        
        # Algorithm config
        algo_cfg = config.algo
        self.stage = getattr(algo_cfg, 'stage', 1)
        self.sampling_eps = getattr(algo_cfg, 'sampling_eps', 1e-3)
        self.loss_type = getattr(algo_cfg, 'loss_type', 'simple')  # 'simple' or 'weighted'
        self.parameterization = getattr(algo_cfg, 'parameterization', 'x0')
        self.objective_name = str(getattr(algo_cfg, 'objective', 'ddpm')).lower()
        self.interpolant_name = str(
            getattr(algo_cfg, 'interpolant', 'vp' if self.objective_name == 'ddpm' else 'rectified')
        ).lower()
        self.conditioning_mode = str(getattr(algo_cfg, 'conditioning', 'none')).lower()
        self.embedding_provider_name = str(getattr(algo_cfg, 'embedding_provider', 'legacy_contextual')).lower()
        self.embedding_trainable = bool(getattr(algo_cfg, 'embedding_trainable', False))
        self.lambda_ce = float(getattr(algo_cfg, 'lambda_ce', 0.0))
        self.train_timesteps = int(getattr(algo_cfg, 'train_timesteps', 1000))
        if self.train_timesteps < 2:
            raise ValueError("algo.train_timesteps must be >= 2")

        self.self_conditioning = bool(
            omegaconf.OmegaConf.select(config, "algo.stabilizers.self_conditioning", default=False)
        )
        self.self_conditioning_prob = float(
            omegaconf.OmegaConf.select(config, "algo.stabilizers.self_conditioning_prob", default=0.5)
        )
        self.anchor_weight = float(
            omegaconf.OmegaConf.select(config, "algo.stabilizers.anchor_weight", default=0.0)
        )
        self.noise_scale = float(
            omegaconf.OmegaConf.select(config, "algo.stabilizers.noise_scale", default=1.0)
        )
        if self.noise_scale <= 0:
            raise ValueError("algo.stabilizers.noise_scale must be > 0")
        self.min_snr_gamma = omegaconf.OmegaConf.select(config, "algo.loss.min_snr_gamma", default=None)

        self.decoder_stage2_enabled = bool(
            omegaconf.OmegaConf.select(config, "algo.decoder.stage2.enabled", default=False)
        )
        if self.stage == 2:
            # Backward compatibility: explicit stage=2 forces legacy decoder-only behavior.
            self.decoder_stage2_enabled = True
        self.decoder_latent_source = getattr(algo_cfg, 'decoder_latent_source', 'teacher_z0')
        self.decoder_timestep_policy = getattr(algo_cfg, 'decoder_timestep_policy', 'uniform_continuous')
        self.decoder_num_steps = int(getattr(algo_cfg, 'decoder_num_steps', 128))
        self.decoder_sampling_eps = float(getattr(algo_cfg, 'decoder_sampling_eps', 1e-5))
        self.decoder_mixed_teacher_prob_start = float(
            getattr(algo_cfg, 'decoder_mixed_teacher_prob_start', 0.5)
        )
        self.decoder_mixed_teacher_prob_end = float(
            getattr(algo_cfg, 'decoder_mixed_teacher_prob_end', 0.0)
        )
        self.antithetic_sampling = getattr(config.training, 'antithetic_sampling', True)

        # iMF curriculum (flow-matching warmup -> iMF ramp).
        self.imf_alpha_target = float(omegaconf.OmegaConf.select(config, "algo.imf.alpha_imf", default=1.0))
        self.imf_warmup_steps = int(omegaconf.OmegaConf.select(config, "algo.imf.warmup_steps", default=0))
        self.imf_ramp_steps = int(omegaconf.OmegaConf.select(config, "algo.imf.ramp_steps", default=0))
        self.imf_alpha_floor = float(omegaconf.OmegaConf.select(config, "algo.imf.alpha_floor", default=0.0))
        if self.imf_warmup_steps < 0 or self.imf_ramp_steps < 0:
            raise ValueError("algo.imf.warmup_steps and algo.imf.ramp_steps must be >= 0")

        if self.decoder_latent_source not in {'teacher_z0', 'predicted_x0', 'mixed'}:
            raise ValueError(
                "algo.decoder_latent_source must be one of "
                "{teacher_z0, predicted_x0, mixed}"
            )
        if self.decoder_timestep_policy not in {'inference_matched', 'uniform_continuous', 'curriculum'}:
            raise ValueError(
                "algo.decoder_timestep_policy must be one of "
                "{inference_matched, uniform_continuous, curriculum}"
            )
        if self.decoder_num_steps <= 0:
            raise ValueError('algo.decoder_num_steps must be positive.')
        if not (0.0 < self.decoder_sampling_eps < 0.5):
            raise ValueError('algo.decoder_sampling_eps must be in (0, 0.5).')
        if self.objective_name not in {'ddpm', 'flow_matching', 'flow_map', 'meanflow', 'imf'}:
            raise ValueError("algo.objective must be one of {ddpm, flow_matching, flow_map, meanflow, imf}")
        if self.interpolant_name not in {'vp', 'rectified'}:
            raise ValueError("algo.interpolant must be one of {vp, rectified}")
        if self.embedding_provider_name not in {'legacy_contextual', 'lookup', 'tied'}:
            raise ValueError("algo.embedding_provider must be one of {legacy_contextual, lookup, tied}")
        
        # Build embedding source (legacy contextual provider or token lookup provider).
        self.encoder = None
        if self.embedding_provider_name == 'legacy_contextual':
            encoder_cfg = getattr(config, 'encoder', None)
            if encoder_cfg is not None and hasattr(encoder_cfg, '_target_'):
                self.encoder = hydra.utils.instantiate(encoder_cfg)
            else:
                encoder_name = getattr(config.model, 'encoder_name', 'Qwen/Qwen3-Embedding-0.6B')
                encoder_dtype = getattr(config.model, 'encoder_dtype', 'bfloat16')
                self.encoder = Qwen3EmbeddingProvider(
                    pretrained_model_name=encoder_name,
                    pooling='token',
                    trainable=False,
                    dtype=encoder_dtype,
                )

            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.embed_dim = int(self.encoder.output_dim)
        else:
            self.embed_dim = int(getattr(config.model, 'embed_dim', 1024))
        
        # Build denoiser — only create heads the objective actually uses so
        # DDP with find_unused_parameters=false doesn't error out.
        model_cfg = config.model
        _needs_velocity_head = self.objective_name in {'flow_matching', 'imf', 'meanflow'}
        _needs_u_head = self.objective_name in {'imf', 'meanflow'}
        self.denoiser = ContinuousDenoiser(
            embed_dim=self.embed_dim,
            hidden_size=getattr(model_cfg, 'hidden_size', self.embed_dim),
            n_heads=getattr(model_cfg, 'n_heads', 16),
            n_blocks=getattr(model_cfg, 'n_blocks', 12),
            mlp_ratio=getattr(model_cfg, 'mlp_ratio', 4),
            dropout=getattr(model_cfg, 'dropout', 0.1),
            time_embed_dim=getattr(model_cfg, 'time_embed_dim', 256),
            parameterization=self.parameterization,
            gradient_checkpointing=getattr(model_cfg, 'gradient_checkpointing', False),
            objective=self.objective_name,
            attn_backend=getattr(model_cfg, 'attn_backend', 'auto'),
            use_velocity_head=_needs_velocity_head,
            use_u_head=_needs_u_head,
        )
        
        # Build decoder
        self.decoder = TransformerDecoder(
            embed_dim=self.embed_dim,
            hidden_size=getattr(model_cfg, 'decoder_hidden_size', self.embed_dim),
            vocab_size=self.vocab_size,
            n_heads=getattr(model_cfg, 'decoder_n_heads', 8),
            n_blocks=getattr(model_cfg, 'decoder_n_blocks', 2),
            dropout=getattr(model_cfg, 'dropout', 0.1),
            gradient_checkpointing=getattr(model_cfg, 'gradient_checkpointing', False),
        )

        if self.embedding_provider_name == 'legacy_contextual':
            self.embedding_provider = ContextualEmbeddingProviderAdapter(
                encoder=self.encoder,
                vocab_size=self.vocab_size,
                dim=self.embed_dim,
                decoder=self.decoder,
            )
        else:
            self.embedding_provider = build_lookup_or_tied_provider(
                provider_name='tied' if self.embedding_provider_name == 'tied' else 'lookup',
                vocab_size=self.vocab_size,
                dim=self.embed_dim,
                trainable=self.embedding_trainable,
            )
        
        # Build noise schedule
        self.noise = hydra.utils.instantiate(config.noise)
        
        # Build forward process
        fp_cfg = getattr(algo_cfg, 'forward_process', None)
        if fp_cfg is not None and hasattr(fp_cfg, '_target_'):
            self._forward_process = hydra.utils.instantiate(fp_cfg, schedule=self.noise)
        else:
            self._forward_process = GaussianForwardProcess(schedule=self.noise)

        # Discrete VP schedule used by centralized DDPM objective/sampler math.
        t_grid = torch.linspace(0.0, 1.0, self.train_timesteps, dtype=torch.float32)
        alphabar = self.noise.alpha_t(t_grid).float().clamp(min=1e-20, max=1.0)
        alphabar = scale_alphabar_noise(alphabar, self.noise_scale)
        alphabar[0] = 1.0
        self.vp_schedule = build_vp_schedule_from_alphabar(alphabar)

        # Optional span-masking conditioning for infilling-style training.
        if self.conditioning_mode == 'span_masking':
            self.span_masker = SpanMasker(
                SpanMaskerConfig(
                    min_spans=int(getattr(algo_cfg, 'span_min_spans', 1)),
                    max_spans=int(getattr(algo_cfg, 'span_max_spans', 3)),
                    length_dist=str(getattr(algo_cfg, 'span_length_dist', 'geometric')),
                    mean_span_length=float(getattr(algo_cfg, 'span_mean_length', 4.0)),
                )
            )
        else:
            self.span_masker = None

        self.objective = self._build_objective()
        
        # Freeze the module that is not trained in the current stage.
        # Must happen before EMA creation so _get_parameters() can inspect this flag.
        self.train_decoder_in_stage1 = (self.lambda_ce > 0.0)
        if self.stage == 1:
            if not self.train_decoder_in_stage1 and self.embedding_provider_name == 'legacy_contextual':
                self._freeze_decoder()
        elif self.stage == 2:
            if self.embedding_provider_name != 'legacy_contextual':
                raise ValueError("Legacy stage-2 decoder training requires embedding_provider=legacy_contextual.")
            self._freeze_denoiser()
        else:
            raise ValueError(f"Unsupported stage: {self.stage}. Expected 1 or 2.")
        
        # EMA (after freeze so _get_parameters sees the right trainable set)
        if config.training.ema > 0:
            self.ema = create_ema(self._get_parameters(), decay=config.training.ema)
        else:
            self.ema = None
        
        # Metrics
        self.metrics = Metrics()
        
        # Training config
        self.lr = config.optim.lr
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self.jvp_failures = 0
        # Default to backend auto-selection unless iMF guardrails request math SDPA.
        set_sdpa_math_mode(False)

    def _build_objective(self):
        if self.objective_name == 'ddpm':
            return DDPMObjective(
                schedule=self.vp_schedule,
                config=DDPMObjectiveConfig(
                    parameterization=self.parameterization,
                    min_snr_gamma=self.min_snr_gamma,
                ),
            )
        if self.objective_name == 'flow_matching':
            return FlowMatchingObjective(
                FlowMatchingObjectiveConfig(
                    t_min=float(getattr(self.config.algo, 't_min', 1e-4)),
                    t_max=float(getattr(self.config.algo, 't_max', 1.0 - 1e-4)),
                )
            )
        if self.objective_name in {'meanflow', 'imf'}:
            return IMFObjective(
                IMFObjectiveConfig(
                    dt_min=float(omegaconf.OmegaConf.select(self.config, 'algo.imf.dt_min', default=1e-3)),
                    t_min=float(getattr(self.config.algo, 't_min', 1e-4)),
                    t_max=float(getattr(self.config.algo, 't_max', 1.0 - 1e-4)),
                    stopgrad_dudt=bool(
                        omegaconf.OmegaConf.select(self.config, 'algo.imf.stopgrad_dudt', default=True)
                    ),
                    use_aux_v_head=bool(
                        omegaconf.OmegaConf.select(self.config, 'algo.imf.use_aux_v_head', default=True)
                    ),
                    alpha_imf=self.imf_alpha_target,
                    fm_aux_weight=float(
                        omegaconf.OmegaConf.select(self.config, 'algo.imf.fm_aux_weight', default=0.05)
                    ),
                )
            )
        if self.objective_name == 'flow_map':
            return FlowMapObjective(
                FlowMapObjectiveConfig(
                    dt_min=float(omegaconf.OmegaConf.select(self.config, 'algo.flow_map.dt_min', default=1e-3)),
                    t_min=float(getattr(self.config.algo, 't_min', 1e-4)),
                    t_max=float(getattr(self.config.algo, 't_max', 1.0 - 1e-4)),
                    s_zero_prob=float(
                        omegaconf.OmegaConf.select(self.config, 'algo.flow_map.s_zero_prob', default=0.0)
                    ),
                )
            )
        raise ValueError(f"Unknown objective: {self.objective_name}")

    def _current_imf_alpha(self) -> float:
        """Compute curriculum-mixed iMF coefficient in [0, 1]."""
        target = float(max(min(self.imf_alpha_target, 1.0), 0.0))
        floor = float(max(min(self.imf_alpha_floor, 1.0), 0.0))
        if target < floor:
            target = floor

        trainer = getattr(self, "_trainer", None)
        if trainer is not None and hasattr(trainer, "global_step"):
            step = int(trainer.global_step)
        else:
            step = int(getattr(self, "global_step", 0))

        if step < self.imf_warmup_steps:
            return floor
        if self.imf_ramp_steps <= 0:
            return target

        progress = min(1.0, max(0.0, (step - self.imf_warmup_steps) / float(self.imf_ramp_steps)))
        return floor + (target - floor) * progress

    def _embed_inputs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        provider_name = getattr(self, 'embedding_provider_name', 'legacy_contextual')
        provider = getattr(self, 'embedding_provider', None)
        if provider is None:
            # Backward-compatible fallback used by lightweight unit tests that
            # instantiate the algorithm without full __init__.
            encoder = getattr(self, 'encoder', None)
            if encoder is None:
                raise AttributeError("No embedding provider/encoder is available for input embedding.")
            with torch.no_grad():
                return encoder(input_ids=input_ids, attention_mask=attention_mask)

        if provider_name == 'legacy_contextual':
            with torch.no_grad():
                return provider.embed(input_ids, attention_mask)
        return provider.embed(input_ids, attention_mask)

    def _decode_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        provider_name = getattr(self, 'embedding_provider_name', 'legacy_contextual')
        provider = getattr(self, 'embedding_provider', None)
        if provider is not None and provider_name in {'lookup', 'tied'}:
            return provider.logits(embeddings)
        return self.decoder(embeddings)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.denoiser(
            z,
            t,
            attention_mask=attention_mask,
            r=r,
            cond=cond,
            self_cond=self_cond,
            span_mask=span_mask,
        )

    def predict_velocity(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.denoiser.predict_velocity(
            z,
            t,
            attention_mask=attention_mask,
            span_mask=span_mask,
            self_cond=self_cond,
        )

    def predict_u(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.denoiser.predict_u(
            z,
            r,
            t,
            attention_mask=attention_mask,
            span_mask=span_mask,
            self_cond=self_cond,
        )
    
    def _freeze_denoiser(self):
        """Freeze denoiser for Stage 2 training."""
        for param in self.denoiser.parameters():
            param.requires_grad = False

    def _freeze_decoder(self):
        """Freeze decoder for Stage 1 denoiser training."""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def to(self, *args, **kwargs):
        """Move model and metrics to device."""
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        return self
    
    def _get_parameters(self):
        """Return trainable parameters based on stage."""
        if self.stage == 2:
            return self.decoder.parameters()
        params = [self.denoiser.parameters()]
        if self.train_decoder_in_stage1 and self.embedding_provider_name == 'legacy_contextual':
            params.append(self.decoder.parameters())
        if self.embedding_provider_name != 'legacy_contextual' and self.embedding_trainable:
            params.append(self.embedding_provider.parameters())
        return itertools.chain(*params)
    
    def _sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps with antithetic sampling."""
        t0 = torch.rand((), device=device)
        t = (t0 + torch.arange(batch_size, device=device, dtype=torch.float32) / max(batch_size, 1)) % 1.0
        return t.clamp(self.sampling_eps, 1.0 - self.sampling_eps)

    def _decoder_inference_timestep_grid(self, device: torch.device) -> torch.Tensor:
        """Discrete timestep grid that matches reverse generation schedule."""
        return torch.linspace(
            1.0 - self.decoder_sampling_eps,
            self.decoder_sampling_eps,
            self.decoder_num_steps + 1,
            device=device,
        )

    def _training_progress_ratio(self) -> float:
        """Approximate training progress in [0, 1] for simple schedules."""
        if self.trainer is None:
            return 0.0
        max_steps = getattr(self.trainer, 'max_steps', None)
        if max_steps is None or int(max_steps) <= 0:
            return 0.0
        return float(min(max(float(self.trainer.global_step) / float(max_steps), 0.0), 1.0))

    def _sample_decoder_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample decoder-training timesteps according to configured policy."""
        if self.decoder_timestep_policy == 'uniform_continuous':
            return self._sample_timesteps(batch_size, device)

        grid = self._decoder_inference_timestep_grid(device)
        candidates = grid[:-1]  # Reverse loop uses all but the final endpoint.

        if self.decoder_timestep_policy == 'curriculum':
            # Start from high-noise steps and progressively unlock cleaner steps.
            progress = self._training_progress_ratio()
            unlocked = max(1, int(round(progress * max(candidates.numel() - 1, 1))) + 1)
            candidates = candidates[:unlocked]

        indices = torch.randint(
            low=0,
            high=int(candidates.numel()),
            size=(batch_size,),
            device=device,
        )
        return candidates.index_select(0, indices)

    def _mixed_teacher_probability(self) -> float:
        """Teacher latent probability for mixed decoder training."""
        progress = self._training_progress_ratio()
        return float(
            self.decoder_mixed_teacher_prob_start
            + (self.decoder_mixed_teacher_prob_end - self.decoder_mixed_teacher_prob_start) * progress
        )

    def _compute_predicted_x0_latents(
        self,
        z_0: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Create denoiser-predicted clean latents from noised encoder latents."""
        batch_size = int(z_0.shape[0])
        t = self._sample_decoder_timesteps(batch_size, z_0.device)
        z_t, _ = self._forward_process(z_0, t)
        alpha_t = self._get_alpha_t(t, z_0.shape)

        with torch.no_grad():
            return self.denoiser.predict_x0(z_t, t, alpha_t, attention_mask)

    def _stage2_decoder_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select decoder training latents for stage-2 CE."""
        z_0 = self._embed_inputs(input_ids, attention_mask)

        if self.decoder_latent_source == 'teacher_z0':
            return z_0

        z_hat_0 = self._compute_predicted_x0_latents(z_0, attention_mask)
        if self.decoder_latent_source == 'predicted_x0':
            return z_hat_0

        # Mixed mode: sample per-example teacher/predicted source.
        teacher_prob = self._mixed_teacher_probability()
        if teacher_prob <= 0.0:
            return z_hat_0
        if teacher_prob >= 1.0:
            return z_0

        use_teacher = torch.rand(z_0.shape[0], device=z_0.device) < teacher_prob
        while use_teacher.ndim < z_0.ndim:
            use_teacher = use_teacher.unsqueeze(-1)
        return torch.where(use_teacher, z_0, z_hat_0)
    
    def _get_alpha_t(self, t: torch.Tensor, z_shape: tuple) -> torch.Tensor:
        """Get alpha_t broadcasted to embedding shape."""
        alpha_t = self.noise.alpha_t(t)
        # Broadcast to [B, 1, 1] for [B, L, D] embeddings
        while alpha_t.ndim < len(z_shape):
            alpha_t = alpha_t.unsqueeze(-1)
        return alpha_t
    
    def _compute_target(
        self,
        z_0: torch.Tensor,
        z_t: torch.Tensor,
        epsilon: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training target based on parameterization."""
        if self.parameterization == 'x0':
            return z_0
        elif self.parameterization == 'epsilon':
            return epsilon
        elif self.parameterization == 'v':
            # v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * z_0
            sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-20))
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-20))
            return sqrt_alpha_t * epsilon - sqrt_one_minus_alpha_t * z_0
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
    
    def _compute_mse_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_prime_t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute MSE loss between prediction and target.
        
        Args:
            prediction: Model prediction [B, L, D]
            target: Target tensor [B, L, D]
            alpha_t: Schedule alpha values [B, 1, 1]
            alpha_prime_t: Schedule alpha derivative [B, 1, 1]
            attention_mask: Optional mask [B, L]
            
        Returns:
            Scalar loss tensor
        """
        # Per-element squared error
        sq_error = (prediction - target) ** 2
        
        # Mean over embedding dimension
        per_token_mse = sq_error.mean(dim=-1)  # [B, L]
        
        if self.loss_type == 'weighted':
            # ELBO-style weighting: weight = -alpha'_t / (1 - alpha_t)
            weight = (-alpha_prime_t / (1.0 - alpha_t).clamp(min=1e-20)).squeeze(-1)  # [B, 1] or [B]
            if weight.ndim == 1:
                weight = weight.unsqueeze(-1)  # [B, 1]
            per_token_mse = per_token_mse * weight
        
        # Apply attention mask if provided
        if attention_mask is not None:
            per_token_mse = per_token_mse * attention_mask.float()
            num_valid = attention_mask.sum().clamp(min=1)
            return per_token_mse.sum() / num_valid
        
        return per_token_mse.mean()
    
    def _compute_ce_loss(
        self,
        z_hat_0: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for decoded tokens.
        
        Args:
            z_hat_0: Predicted clean embeddings [B, L, D]
            input_ids: Target token IDs [B, L]
            attention_mask: Optional mask [B, L]
            
        Returns:
            Scalar CE loss tensor
        """
        logits = self._decode_logits(z_hat_0)  # [B, L, V]
        
        # Flatten for cross-entropy
        B, L, V = logits.shape
        flat_logits = logits.view(-1, V)
        flat_targets = input_ids.view(-1)
        
        # Per-token CE
        per_token_ce = F.cross_entropy(flat_logits, flat_targets, reduction='none').view(B, L)
        
        mask = None
        if attention_mask is not None:
            mask = attention_mask.bool()
        if token_mask is not None:
            mask = token_mask.bool() if mask is None else (mask & token_mask.bool())
        if mask is not None:
            per_token_ce = per_token_ce * mask.float()
            num_valid = mask.sum().clamp(min=1)
            return per_token_ce.sum() / num_valid
        
        return per_token_ce.mean()
    
    def _loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        current_accumulation_step: Optional[int] = None,
        train_mode: bool = False,
    ) -> Loss:
        """Compute training loss.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            current_accumulation_step: Current gradient accumulation step
            train_mode: Whether in training mode
            
        Returns:
            Loss dataclass with all loss components
        """
        B, L = input_ids.shape
        device = input_ids.device
        del current_accumulation_step, train_mode

        if self.stage == 1:
            z_0 = self._embed_inputs(input_ids, attention_mask)  # [B, L, D]
            span_mask = None
            if self.span_masker is not None:
                span_mask, _ = self.span_masker.make_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=getattr(self.tokenizer, 'mask_token_id', None),
                )

            if self.objective_name in {'imf', 'meanflow'} and hasattr(self.objective, "config"):
                self.objective.config.alpha_imf = self._current_imf_alpha()

            objective_out = self.objective.compute(
                model=self,
                x0=z_0,
                attention_mask=attention_mask,
                span_mask=span_mask,
                input_ids=input_ids,
            )
            mse_loss = objective_out.loss
            aux_losses = {k: v for k, v in (objective_out.aux_losses or {}).items()}
            ce_loss = torch.tensor(0.0, device=device)
            anchor_loss = torch.tensor(0.0, device=device)
            if self.lambda_ce > 0.0:
                ce_loss = self._compute_ce_loss(
                    objective_out.x0_hat if objective_out.x0_hat is not None else z_0,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_mask=span_mask,
                )
            total_loss = mse_loss + self.lambda_ce * ce_loss
            if self.anchor_weight > 0.0 and objective_out.x0_hat is not None:
                anchor_sq = (objective_out.x0_hat - z_0.detach()) ** 2
                if attention_mask is not None:
                    valid = attention_mask.bool()
                    if span_mask is not None:
                        valid = valid & span_mask.bool()
                    while valid.ndim < anchor_sq.ndim:
                        valid = valid.unsqueeze(-1)
                    anchor_sq = anchor_sq * valid.float()
                    denom = valid.sum().clamp(min=1) * anchor_sq.shape[-1]
                    anchor_loss = anchor_sq.sum() / denom
                else:
                    anchor_loss = anchor_sq.mean()
                total_loss = total_loss + self.anchor_weight * anchor_loss
                aux_losses["anchor"] = anchor_loss
        else:
            # Stage 2: decoder-only objective with configurable latent source.
            decoder_latents = self._stage2_decoder_latents(input_ids, attention_mask)
            ce_loss = self._compute_ce_loss(decoder_latents, input_ids, attention_mask)
            mse_loss = torch.tensor(0.0, device=device)
            total_loss = ce_loss
            aux_losses = None
        
        num_tokens = attention_mask.sum()
        
        return Loss(
            loss=total_loss,
            nlls=total_loss * num_tokens,
            num_tokens=num_tokens,
            mse_loss=mse_loss,
            ce_loss=ce_loss,
            aux_losses=aux_losses,
        )
    
    # Lightning hooks
    def on_train_epoch_start(self):
        self.metrics.reset()
    
    def training_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        losses = self._loss(
            batch['input_ids'],
            batch['attention_mask'],
            current_accumulation_step,
            train_mode=True,
        )
        
        self.metrics.update_train(losses.nlls, losses.num_tokens)
        
        # Logging
        self.log('trainer/loss', losses.loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        self.log('trainer/mse_loss', losses.mse_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log('trainer/ce_loss', losses.ce_loss, on_step=True, on_epoch=False, sync_dist=True)
        if self.objective_name in {'imf', 'meanflow'} and self.stage == 1:
            self.log(
                'trainer/alpha_imf',
                torch.tensor(self._current_imf_alpha(), device=losses.loss.device),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        if losses.aux_losses:
            for key, value in losses.aux_losses.items():
                self.log(f'trainer/{key}_loss', value, on_step=True, on_epoch=False, sync_dist=True)
        if self.stage == 2 and self.decoder_latent_source == 'mixed':
            self.log(
                'trainer/decoder_teacher_prob',
                torch.tensor(self._mixed_teacher_probability(), device=losses.loss.device),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        
        return losses.loss
    
    def on_train_epoch_end(self):
        train_metrics = {}
        for k, v in self.metrics.train_nlls.items():
            if getattr(v, 'weight', 0) > 0:
                train_metrics[k] = v.compute()
        if train_metrics:
            self.log_dict(train_metrics, on_step=False, on_epoch=True, sync_dist=True)
    
    def on_validation_epoch_start(self):
        self.metrics.reset()
        self._eval_mode()
    
    def validation_step(self, batch, batch_idx):
        losses = self._loss(batch['input_ids'], batch['attention_mask'])
        self.metrics.update_valid(losses.nlls, losses.num_tokens)
        
        self.log('val/loss', losses.loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/mse_loss', losses.mse_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/ce_loss', losses.ce_loss, on_step=False, on_epoch=True, sync_dist=True)
        if losses.aux_losses:
            for key, value in losses.aux_losses.items():
                self.log(f'val/{key}_loss', value, on_step=False, on_epoch=True, sync_dist=True)
        
        return losses.loss
    
    def on_validation_epoch_end(self):
        valid_metrics = {}
        for k, v in self.metrics.valid_nlls.items():
            if getattr(v, 'weight', 0) > 0:
                valid_metrics[k] = v.compute()
        if valid_metrics:
            self.log_dict(valid_metrics, on_step=False, on_epoch=True, sync_dist=True)
        if ((self.config.eval.compute_perplexity_on_sanity or not self.trainer.sanity_checking)
                and getattr(self.config.eval, 'generate_samples', False)):
            try:
                samples = self.generate_samples(num_samples=self.config.loader.eval_batch_size)
                if getattr(self.config.eval, 'save_validation_samples', False):
                    save_dir = Path(os.getcwd()) / 'validation_samples'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f'step_{self.global_step}.pt'
                    torch.save(samples.detach().cpu(), save_path.as_posix())
            except Exception as exc:
                print(f"Sampling failed at step {self.global_step}: {exc}")
        self._train_mode()
    
    def _eval_mode(self):
        """Switch to evaluation mode with EMA weights."""
        if self.ema:
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        self.denoiser.eval()
        self.decoder.eval()
        if isinstance(self.embedding_provider, nn.Module):
            self.embedding_provider.eval()
    
    def _train_mode(self):
        """Switch back to training mode."""
        if self.ema:
            self.ema.restore(self._get_parameters())
        if self.stage == 1:
            self.denoiser.train()
            if self.train_decoder_in_stage1 and self.embedding_provider_name == 'legacy_contextual':
                self.decoder.train()
            else:
                self.decoder.eval()
            if isinstance(self.embedding_provider, nn.Module):
                if self.embedding_provider_name != 'legacy_contextual' and self.embedding_trainable:
                    self.embedding_provider.train()
                else:
                    self.embedding_provider.eval()
        else:
            self.denoiser.eval()
            self.decoder.train()
            if isinstance(self.embedding_provider, nn.Module):
                self.embedding_provider.eval()
    
    def on_train_start(self):
        smoke = run_sampler_smoke(device=self.device)
        if not smoke.ok:
            raise RuntimeError(
                "Continuous sampler smoke-test failed before training start. "
                f"Reason: {smoke.message}"
            )
        if self.objective_name in {'imf', 'meanflow'}:
            set_sdpa_math_mode(False)
            probe = probe_model_jvp(self)
            if not probe.ok:
                self.jvp_failures += 1
                if hasattr(self.denoiser, "set_attention_backend"):
                    warnings.warn(
                        "iMF JVP probe failed; switching denoiser attention backend to "
                        f"'sdpa' with math-only SDPA kernels. Probe details: {probe.message}",
                        RuntimeWarning,
                    )
                    self.denoiser.set_attention_backend("sdpa")
                    set_sdpa_math_mode(True)
                    probe = probe_model_jvp(self)
            if not probe.ok:
                self.jvp_failures += 1
                raise RuntimeError(
                    "iMF JVP probe failed after switching to JVP-safe attention backend. "
                    f"Details: {probe.message}"
                )
            self.log(
                "trainer/jvp_failures",
                torch.tensor(float(self.jvp_failures), device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())
    
    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
    
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )
        
        scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]
    
    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int,
        num_steps: int = 50,
        seq_len: Optional[int] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Generate samples using the configured continuous sampler.
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps
            seq_len: Sequence length (defaults to config)
            eps: Small value to avoid numerical issues
            
        Returns:
            Generated token IDs [num_samples, seq_len]
        """
        sampler_cfg = getattr(self.config.algo, "sampler", None)
        if sampler_cfg is None or not getattr(sampler_cfg, "_target_", None):
            sampler_cfg = getattr(self.config, "sampling", None)
        if sampler_cfg is None or not getattr(sampler_cfg, "_target_", None):
            raise RuntimeError("No sampler configured for ContinuousEmbeddingDiffusion.")

        sampler = hydra.utils.instantiate(
            sampler_cfg,
            self.config,
            forward_process=getattr(self, "_forward_process", None),
            _recursive_=False,
        )
        return sampler.generate(
            model=self,
            num_samples=num_samples,
            num_steps=num_steps,
            eps=eps,
            inject_bos=False,
            seq_len=seq_len,
        )


__all__ = ['ContinuousEmbeddingDiffusion']
