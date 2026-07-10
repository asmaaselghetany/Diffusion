"""Per-token ELBO terms for block masked / uniform training."""

from __future__ import annotations

import torch


def masked_block_nll_per_token(
    log_x_theta: torch.Tensor,
    x0: torch.Tensor,
    alpha_t: torch.Tensor,
    dalpha_t: torch.Tensor,
) -> torch.Tensor:
  log_p = torch.gather(log_x_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
  coeff = dalpha_t / (1 - alpha_t)
  return coeff * log_p


def uniform_block_nll_per_token(
    log_x_theta: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    alpha_t: torch.Tensor,
    dalpha_t: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
  assert alpha_t.ndim == 2
  x_reconst = log_x_theta.exp()
  alpha = alpha_t.unsqueeze(-1)
  x_bar_theta = vocab_size * alpha * x_reconst + 1 - alpha
  coeff = dalpha_t / (vocab_size * alpha_t.squeeze(-1))

  x_eq_xt = (x0 == xt).float()
  x_neq_xt = 1 - x_eq_xt
  xbar_xt = (1 - alpha.squeeze(-1)) + vocab_size * alpha.squeeze(-1) * x_eq_xt
  xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
  xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)

  term1 = vocab_size * (1 / xbar_xt - 1 / xbar_theta_xt)
  const = (1 - alpha.squeeze(-1)) / (
      vocab_size * alpha.squeeze(-1) + 1 - alpha.squeeze(-1))
  term2_coefs = x_eq_xt * const + x_neq_xt
  term2_offset = (
      (vocab_size - 1) * const * x_eq_xt - (1 / const) * x_neq_xt) * const.log()
  term2_theta = -term2_coefs * (
      x_bar_theta.log().sum(-1) - vocab_size * xbar_theta_xt.log())
  term2_theta = term2_theta - vocab_size * alpha.squeeze(-1) / (
      1 - alpha.squeeze(-1)) * (
          xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt
  term2 = term2_theta + term2_offset
  return coeff * (term1 - term2)


def subs_log_probs(
    logits: torch.Tensor,
    xt: torch.Tensor,
    mask_id: int,
    neg_infinity: float,
) -> torch.Tensor:
  logits = logits.clone()
  logits[..., mask_id] = neg_infinity
  unmasked = xt != mask_id
  logits[unmasked] = neg_infinity
  logits[unmasked, xt[unmasked]] = 0.0
  return logits - torch.logsumexp(logits, dim=-1, keepdim=True)
