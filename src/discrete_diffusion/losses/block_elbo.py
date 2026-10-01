"""Per-token ELBO terms for block masked / uniform training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_plain_ce_per_token(
    log_x_theta: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    mask_id: int,
) -> torch.Tensor:
  """Unweighted masked-token CE (Fast-dLLM complementary objective).

  Returns per-token CE with zeros on clean sites. Callers that mean-reduce
  (``BlockTrainer._loss`` with ``loss_weighting=plain_ce``) MUST divide by
  the count of mask sites only — Hub ``ForCausalLMLoss`` ignores ``labels=-100``
  on clean tokens. Dividing by all valid tokens under-scales vs Hub (~½ with
  complementary 2B).
  """
  log_probs = F.log_softmax(log_x_theta, dim=-1)
  ce = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
  return (xt == mask_id).to(ce.dtype) * ce


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
    mask_id: int | None = None,
    exclude_ids: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
  """DUO/UDLM per-token uniform ELBO.

  ``vocab_size`` is the **simplex** size ``V_eff`` used in all coefficients
  (BlockGen/Duo: full tokenizer ``V``; our Qwen conversion: ``V-|E|`` when
  reserved specials sit inside the embedding table).

  When ``log_x_theta`` still has width ``V > V_eff`` with banned specials
  (``p≈0`` on those slots), each dead slot of ``x̄_θ = V_eff·α·p+(1-α)``
  equals ``1-α``. Summing ``log x̄_θ`` over the full width then injects
  extra ``log(1-α)`` terms Duo/BlockGen never have. Pass ``mask_id`` and/or
  ``exclude_ids`` to subtract those dead slots from the sum.
  """
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
  sum_log_xbar = x_bar_theta.log().sum(-1)
  dead: list[int] = []
  if mask_id is not None:
    dead.append(int(mask_id))
  if exclude_ids is not None:
    dead.extend(int(x) for x in exclude_ids)
  # Unique, in-range dead slots under a padded logit width.
  seen: set[int] = set()
  width = x_bar_theta.size(-1)
  if width > vocab_size:
    for mid in dead:
      if mid in seen or not (0 <= mid < width):
        continue
      seen.add(mid)
      sum_log_xbar = sum_log_xbar - x_bar_theta[..., mid].clamp_min(1e-12).log()
  term2_theta = -term2_coefs * (
      sum_log_xbar - vocab_size * xbar_theta_xt.log())
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
