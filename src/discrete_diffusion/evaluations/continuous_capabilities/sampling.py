"""Latent-space sampling helpers for continuous capability evaluation."""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import torch

from ...continuous.math import (
    build_vp_schedule_from_alphabar,
    scale_alphabar_noise,
    eps_from_x0,
    q_posterior,
)
from ...sampling.token_selection import (
    GreedySelection,
    NucleusSelection,
    TemperatureSelection,
    TopKSelection,
)


SamplingMethod = Literal["ddpm", "ddim", "ancestral"]



def _alpha_t(model, t: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    alpha = model.noise.alpha_t(t)
    while alpha.ndim < len(target_shape):
        alpha = alpha.unsqueeze(-1)
    return alpha



def apply_fixed_latent_conditioning(
    z_t: torch.Tensor,
    z_known: torch.Tensor,
    fixed_mask: torch.Tensor,
    alpha_t: torch.Tensor,
    fixed_noise: torch.Tensor,
) -> torch.Tensor:
    """Project fixed positions onto q(z_t | z_known) with a shared noise draw."""
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    conditioned = sqrt_alpha_t * z_known + sqrt_one_minus_alpha_t * fixed_noise
    return torch.where(fixed_mask.unsqueeze(-1), conditioned, z_t)



def _predict_z0(
    model,
    z_t: torch.Tensor,
    t: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    clip_denoised: bool,
    clip_range: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha_t = _alpha_t(model, t, z_t.shape)
    z_hat_0 = model.denoiser.predict_x0(z_t, t, alpha_t, attention_mask)
    if clip_denoised:
        z_hat_0 = z_hat_0.clamp(clip_range[0], clip_range[1])
    return z_hat_0, alpha_t



def _ddpm_step(
    model,
    z_t: torch.Tensor,
    t_model: torch.Tensor,
    t_idx: torch.Tensor,
    t_prev_idx: torch.Tensor,
    vp_schedule,
    *,
    attention_mask: Optional[torch.Tensor],
    clip_denoised: bool,
    clip_range: Tuple[float, float],
) -> torch.Tensor:
    z_hat_0, alpha_t = _predict_z0(
        model,
        z_t,
        t_model,
        attention_mask=attention_mask,
        clip_denoised=clip_denoised,
        clip_range=clip_range,
    )
    del alpha_t
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
    model,
    z_t: torch.Tensor,
    t_model: torch.Tensor,
    t_idx: torch.Tensor,
    t_prev_idx: torch.Tensor,
    vp_schedule,
    *,
    eta: float,
    attention_mask: Optional[torch.Tensor],
    clip_denoised: bool,
    clip_range: Tuple[float, float],
) -> torch.Tensor:
    z_hat_0, alpha_t = _predict_z0(
        model,
        z_t,
        t_model,
        attention_mask=attention_mask,
        clip_denoised=clip_denoised,
        clip_range=clip_range,
    )
    del alpha_t
    epsilon = eps_from_x0(x_t=z_t, x0=z_hat_0, t=t_idx, sched=vp_schedule)

    ab_t = vp_schedule.alphabar.index_select(0, t_idx.view(-1)).to(device=z_t.device, dtype=z_t.dtype)
    ab_prev = vp_schedule.alphabar.index_select(0, t_prev_idx.view(-1)).to(device=z_t.device, dtype=z_t.dtype)
    while ab_t.ndim < z_t.ndim:
        ab_t = ab_t.unsqueeze(-1)
        ab_prev = ab_prev.unsqueeze(-1)

    sigma = eta * torch.sqrt(
        ((1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-20))
        * (1.0 - (ab_t / ab_prev.clamp(min=1e-20)))
    )

    pred_direction = torch.sqrt((1.0 - ab_prev - sigma**2).clamp(min=0.0)) * epsilon
    z_t_prev = torch.sqrt(ab_prev.clamp(min=1e-20)) * z_hat_0 + pred_direction

    if eta > 0 and int(t_prev_idx.max().item()) > 0:
        z_t_prev = z_t_prev + sigma * torch.randn_like(z_t)

    return z_t_prev



def _ancestral_step(
    model,
    z_t: torch.Tensor,
    t_model: torch.Tensor,
    t_idx: torch.Tensor,
    t_prev_idx: torch.Tensor,
    vp_schedule,
    *,
    eta: float,
    attention_mask: Optional[torch.Tensor],
    clip_denoised: bool,
    clip_range: Tuple[float, float],
) -> torch.Tensor:
    z_hat_0, alpha_t = _predict_z0(
        model,
        z_t,
        t_model,
        attention_mask=attention_mask,
        clip_denoised=clip_denoised,
        clip_range=clip_range,
    )
    del alpha_t
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
        z_t_prev = z_t_prev + eta * noise_scale * torch.randn_like(z_t)

    return z_t_prev



def _reverse_step(
    model,
    z_t: torch.Tensor,
    t_model: torch.Tensor,
    t_idx: torch.Tensor,
    t_prev_idx: torch.Tensor,
    vp_schedule,
    *,
    sampling_method: SamplingMethod,
    eta: float,
    attention_mask: Optional[torch.Tensor],
    clip_denoised: bool,
    clip_range: Tuple[float, float],
) -> torch.Tensor:
    if sampling_method == "ddpm":
        return _ddpm_step(
            model,
            z_t,
            t_model,
            t_idx,
            t_prev_idx,
            vp_schedule,
            attention_mask=attention_mask,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
        )
    if sampling_method == "ddim":
        return _ddim_step(
            model,
            z_t,
            t_model,
            t_idx,
            t_prev_idx,
            vp_schedule,
            eta=eta,
            attention_mask=attention_mask,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
        )
    if sampling_method == "ancestral":
        return _ancestral_step(
            model,
            z_t,
            t_model,
            t_idx,
            t_prev_idx,
            vp_schedule,
            eta=eta,
            attention_mask=attention_mask,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
        )
    raise ValueError(f"Unknown sampling method: {sampling_method}")



def sample_latents_unconditional(
    model,
    *,
    num_samples: int,
    seq_len: int,
    num_steps: int,
    sampling_method: SamplingMethod,
    eta: float,
    eps: float,
    clip_denoised: bool,
    clip_range: Tuple[float, float],
) -> torch.Tensor:
    """Sample latent trajectories from Gaussian noise."""
    device = next(model.denoiser.parameters()).device
    z_t = torch.randn(num_samples, seq_len, model.embed_dim, device=device)

    timesteps = torch.linspace(1.0 - eps, eps, num_steps + 1, device=device)
    timesteps_fwd = torch.flip(timesteps, dims=[0])
    alphabar_fwd = model.noise.alpha_t(timesteps_fwd).float().clamp(min=1e-20, max=1.0)
    noise_scale = float(getattr(model, "noise_scale", 1.0))
    alphabar_fwd = scale_alphabar_noise(alphabar_fwd, noise_scale)
    vp_schedule = build_vp_schedule_from_alphabar(alphabar_fwd)
    for step_idx in range(num_steps):
        t_model = timesteps[step_idx].expand(num_samples)
        k = num_steps - step_idx
        k_prev = max(k - 1, 0)
        t_idx = torch.full((num_samples,), k, device=device, dtype=torch.long)
        t_prev_idx = torch.full((num_samples,), k_prev, device=device, dtype=torch.long)
        z_t = _reverse_step(
            model,
            z_t,
            t_model,
            t_idx,
            t_prev_idx,
            vp_schedule,
            sampling_method=sampling_method,
            eta=eta,
            attention_mask=None,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
        )

    return z_t



def sample_latents_conditioned(
    model,
    *,
    conditioning_ids: torch.Tensor,
    fixed_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_steps: int,
    sampling_method: SamplingMethod,
    eta: float,
    eps: float,
    clip_denoised: bool,
    clip_range: Tuple[float, float],
) -> torch.Tensor:
    """Sample latents with fixed-position conditioning in latent space."""
    device = conditioning_ids.device
    batch_size, seq_len = conditioning_ids.shape

    if attention_mask is None:
        attention_mask = torch.ones_like(conditioning_ids, device=device)

    with torch.no_grad():
        if hasattr(model, "_embed_inputs"):
            z_known = model._embed_inputs(conditioning_ids, attention_mask)
        elif hasattr(model, "embedding_provider"):
            z_known = model.embedding_provider.embed(conditioning_ids, attention_mask)
        else:
            z_known = model.encoder(conditioning_ids, attention_mask)

    fixed_noise = torch.randn_like(z_known)
    z_t = torch.randn(batch_size, seq_len, model.embed_dim, device=device)

    timesteps = torch.linspace(1.0 - eps, eps, num_steps + 1, device=device)
    timesteps_fwd = torch.flip(timesteps, dims=[0])
    alphabar_fwd = model.noise.alpha_t(timesteps_fwd).float().clamp(min=1e-20, max=1.0)
    noise_scale = float(getattr(model, "noise_scale", 1.0))
    alphabar_fwd = scale_alphabar_noise(alphabar_fwd, noise_scale)
    vp_schedule = build_vp_schedule_from_alphabar(alphabar_fwd)
    alpha_start = _alpha_t(model, timesteps[0].expand(batch_size), z_t.shape)
    z_t = apply_fixed_latent_conditioning(z_t, z_known, fixed_mask, alpha_start, fixed_noise)

    for step_idx in range(num_steps):
        t_model = timesteps[step_idx].expand(batch_size)
        k = num_steps - step_idx
        k_prev = max(k - 1, 0)
        t_idx = torch.full((batch_size,), k, device=device, dtype=torch.long)
        t_prev_idx = torch.full((batch_size,), k_prev, device=device, dtype=torch.long)

        z_t = _reverse_step(
            model,
            z_t,
            t_model,
            t_idx,
            t_prev_idx,
            vp_schedule,
            sampling_method=sampling_method,
            eta=eta,
            attention_mask=attention_mask,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
        )

        alpha_prev = _alpha_t(model, timesteps[step_idx + 1].expand(batch_size), z_t.shape)
        z_t = apply_fixed_latent_conditioning(z_t, z_known, fixed_mask, alpha_prev, fixed_noise)

    # Guardrail: known positions must remain pinned to their conditioned trajectory.
    alpha_final = _alpha_t(model, timesteps[-1].expand(batch_size), z_t.shape)
    expected_known = torch.sqrt(alpha_final) * z_known + torch.sqrt(1.0 - alpha_final) * fixed_noise
    m = fixed_mask.bool().unsqueeze(-1)
    known_drift = (z_t - expected_known).abs().masked_fill(~m, 0.0).max()
    if known_drift.item() > 1e-5:
        raise RuntimeError(
            "Conditioned sampler modified fixed positions beyond tolerance: "
            f"{known_drift.item():.3e}"
        )

    return z_t



def _build_selector(top_p: float, top_k: int, temperature: float):
    if top_p < 1.0:
        return NucleusSelection(p=top_p, temperature=temperature)
    if top_k > 0:
        return TopKSelection(k=top_k, temperature=temperature)
    if temperature != 1.0:
        return TemperatureSelection(temperature=temperature)
    return GreedySelection()



def _decode_with_lm_guidance(
    logits: torch.Tensor,
    *,
    selector,
    lm_guidance_model: Any,
    lm_guidance_scale: float,
    lm_guidance_start_token_id: int,
    lm_guidance_mode: Literal["static", "confidence"] = "static",
    lm_guidance_min_scale: float = 0.0,
) -> torch.Tensor:
    """Decode with a causal-LM prior added to per-position decoder logits."""
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits [B, L, V], got {tuple(logits.shape)}")

    if lm_guidance_scale <= 0:
        return selector(logits)

    batch_size, seq_len, vocab_size = logits.shape
    decoder_device = logits.device
    lm_device = next(lm_guidance_model.parameters()).device

    start_ids = torch.full(
        (batch_size, 1),
        int(lm_guidance_start_token_id),
        dtype=torch.long,
        device=lm_device,
    )

    with torch.no_grad():
        lm_out = lm_guidance_model(input_ids=start_ids, use_cache=True)
        lm_next_logits = lm_out.logits[:, -1, :]
        past_key_values = lm_out.past_key_values

        lm_vocab_size = int(lm_next_logits.shape[-1])

        # Most checkpoints use GPT-2 vocab plus a small number of added specials.
        # Keep guidance on overlapping IDs and leave non-overlap IDs unguided.
        if lm_vocab_size == vocab_size:
            overlap_vocab_size = vocab_size
        else:
            overlap_vocab_size = min(vocab_size, lm_vocab_size)
            if overlap_vocab_size <= 0:
                raise ValueError(
                    "LM guidance vocab mismatch with no overlap: "
                    f"decoder_vocab={vocab_size}, lm_vocab={lm_vocab_size}"
                )

        sampled_tokens = []
        for pos in range(seq_len):
            decoder_logits = logits[:, pos, :].float()
            lm_logits_pos = torch.zeros_like(decoder_logits)
            lm_logits_pos[:, :overlap_vocab_size] = lm_next_logits[:, :overlap_vocab_size].to(
                decoder_device,
                dtype=torch.float32,
            )

            if lm_guidance_mode == "confidence":
                decoder_probs = torch.softmax(decoder_logits, dim=-1)
                decoder_confidence = decoder_probs.max(dim=-1).values
                # Increase LM guidance when decoder is uncertain.
                dynamic_scale = float(lm_guidance_min_scale) + (
                    float(lm_guidance_scale) - float(lm_guidance_min_scale)
                ) * (1.0 - decoder_confidence)
                combined_logits = decoder_logits + dynamic_scale.unsqueeze(-1) * lm_logits_pos
            else:
                combined_logits = decoder_logits + float(lm_guidance_scale) * lm_logits_pos

            next_tokens = selector(combined_logits)
            sampled_tokens.append(next_tokens)

            lm_out = lm_guidance_model(
                input_ids=next_tokens.to(lm_device).unsqueeze(-1),
                past_key_values=past_key_values,
                use_cache=True,
            )
            lm_next_logits = lm_out.logits[:, -1, :]
            past_key_values = lm_out.past_key_values

    return torch.stack(sampled_tokens, dim=1)


def decode_embeddings(
    model,
    tokenizer,
    embeddings: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    ban_special_tokens: bool,
    additional_banned_token_ids: Optional[list[int]] = None,
    lm_guidance_model: Optional[Any] = None,
    lm_guidance_scale: float = 0.0,
    lm_guidance_start_token_id: Optional[int] = None,
    lm_guidance_mode: Literal["static", "confidence"] = "static",
    lm_guidance_min_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode latent embeddings to token IDs."""
    if hasattr(model, "embedding_provider") and getattr(model, "embedding_provider_name", "legacy_contextual") in {"lookup", "tied"}:
        logits = model.embedding_provider.logits(embeddings)
    else:
        logits = model.decoder(embeddings)
    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)

    if ban_special_tokens:
        banned = []
        if getattr(tokenizer, "pad_token_id", None) is not None:
            banned.append(int(tokenizer.pad_token_id))
        if getattr(tokenizer, "mask_token_id", None) is not None:
            banned.append(int(tokenizer.mask_token_id))
        if additional_banned_token_ids:
            banned.extend(int(tok_id) for tok_id in additional_banned_token_ids)
        if banned:
            logits[..., sorted(set(banned))] = -float("inf")

    selector = _build_selector(top_p=top_p, top_k=top_k, temperature=temperature)
    if lm_guidance_model is not None and float(lm_guidance_scale) > 0:
        start_token_id = (
            int(lm_guidance_start_token_id)
            if lm_guidance_start_token_id is not None
            else int(getattr(tokenizer, "eos_token_id", 0) or 0)
        )
        token_ids = _decode_with_lm_guidance(
            logits,
            selector=selector,
            lm_guidance_model=lm_guidance_model,
            lm_guidance_scale=float(lm_guidance_scale),
            lm_guidance_start_token_id=start_token_id,
            lm_guidance_mode=lm_guidance_mode,
            lm_guidance_min_scale=float(lm_guidance_min_scale),
        )
    else:
        token_ids = selector(logits)
    return token_ids, logits


__all__ = [
    "SamplingMethod",
    "apply_fixed_latent_conditioning",
    "sample_latents_unconditional",
    "sample_latents_conditioned",
    "decode_embeddings",
]
