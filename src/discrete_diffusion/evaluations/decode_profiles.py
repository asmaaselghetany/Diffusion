"""Shared decode profiles + free-sample modes for block_qwen eval.

Single source of truth for:
  - ``DECODE_PROFILES`` (baseline / hierarchical / hierarchical_ancestral /
    hierarchical_ancestral_t01 / hierarchical_ss / hierarchical_ss_ancestral /
    hierarchical_quiet / hubmatch / dual_cache / uniform_dual / uniform_dual_random /
    ss_quiet_ancestral / uniform_commit / uniform_commit_t1 / uniform_commit_ss)
  - sample_mode inference (native_free vs conversion_free)
  - samples.meta.json schema / reuse gate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from discrete_diffusion.data.conversion_baseline import (
    apply_conversion_chat_template,
    conversion_prefix_text as _conversion_prefix_text,
)


# Shared decode profiles. Keep lm-eval / throughput / free-gen in sync.
#
# Conversion floor name = ``hierarchical`` (``baseline`` coerces here).
# Mid-α ancestral at T=1 is ~2–5% Instruct GSM (forensic; hard twins 2.4/5.4).
# That shows masked needs remask — not that Unif collapse is "just knobs"
# (T=0.1 / ss cells pending). Matched gap baseline floor = ARPC; UCC is a
# homemade probe. Remask (thr / DualCache / ARPC) or UCC = usable *products*.
# ``baseline`` ≡ ``hierarchical`` on every arm. UCC is **only**
# ``uniform_commit`` / ``uniform_dual`` / ``uniform_commit_t1`` — never coerced
# onto hierarchical / hubmatch / dual_cache.
# Forensic mid-α: ``hierarchical_ancestral`` (+ T / packing twins below).
#
# Levers: hubmatch / dual_cache / quiet / ARPC / ancestral; UCC = UC cell only.
# Full-seq dual only via ``full_seq_dual`` + ``allow_full_seq_decode``.
DECODE_PROFILES: dict[str, list[str]] = {
    # Identical pins to ``hierarchical`` (BlockGen open-loop skeleton).
    'baseline': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=0.9',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # Legacy ablation: full-seq dual + x0[active]=xt (C0 ancestral 2.27% path).
    # Requires sampling.allow_full_seq_decode=true / ALLOW_FULL_SEQ_DECODE=1.
    'full_seq_dual': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=false',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=true',
        # Clear sticky UCC if a train baked sticky=true into the ckpt yaml.
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # Conversion floor for BOTH arms (``baseline`` coerces here).
    # Remask/commit: thr=0.9 + greedy. Mid-α ancestral thr=null is
    # ``hierarchical_ancestral`` (forensic only; ~2% Instruct GSM).
    'hierarchical': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=0.9',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=1.0',
        # BlockGen-parity uniform posterior (ignored on masked).
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        # Clear UCC if a train baked sticky=true into the ckpt yaml.
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],

    # Mid-α Bernoulli ancestral (no thr). T=1 pack-as-is cell of Tier-2 2×2.
    # ~2–5% GSM historically — T confound pending ancestral_t01.
    'hierarchical_ancestral': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # Matched T=0.1 twin of ``hierarchical_ancestral`` — packing identical;
    # only x0_temperature changes (no ARPC). Isolates quiet-T vs collapse.
    'hierarchical_ancestral_t01': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=0.1',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # Tier-2 pack column: Hub ss+sub8 + mid-α ancestral @ T=1.
    # sub=8 matches ``ss_quiet_ancestral`` so only T differs across that row.
    'hierarchical_ss_ancestral': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=8',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
        'sampling.uniform_commit_random=false',
    ],
    # Hub single-stream packing + confidence (packing ablation of floor).
    'hierarchical_ss': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=0.9',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=null',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # B1 packing + BlockGen ARPC (re-MASK ∥ re-Unif). Not TinyGSM quiet T.
    'hierarchical_arpc': [
        'sampling.use_arpc=true',
        'sampling.arpc_mode=blockgen',
        'sampling.arpc_corruption_mode=ar_metric',
        'sampling.arpc_ar_metric=nll',
        'sampling.arpc_use_prefix_fill=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=1.0',
        'sampling.arpc_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # Matched T=0.1 twin of ``hierarchical_arpc`` — packing/scorer/rounds
    # identical; only x0_temperature + arpc_temperature change. Use this for
    # the ARPC∥ARPC baseline T=0.1 row (NOT ``hierarchical_quiet``, which also
    # flips single_stream_decode).
    'hierarchical_arpc_t01': [
        'sampling.use_arpc=true',
        'sampling.arpc_mode=blockgen',
        'sampling.arpc_corruption_mode=ar_metric',
        'sampling.arpc_ar_metric=nll',
        'sampling.arpc_use_prefix_fill=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=0.1',
        'sampling.arpc_temperature=0.1',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # BlockGen-style quieter ancestral: single-stream + low T + ARPC.
    # TinyGSM uses one TEMP≈0.1 on logits for the whole AR-then-ARPC path.
    # ARPC on ckpts without size-1 mixture still runs (sampler warns).
    # Secondary / TinyGSM packing probe — NOT the matched T=0.1 baseline row.
    'hierarchical_quiet': [
        'sampling.use_arpc=true',
        'sampling.arpc_mode=blockgen',
        'sampling.arpc_corruption_mode=ar_metric',
        'sampling.arpc_ar_metric=nll',
        'sampling.arpc_use_prefix_fill=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=null',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=0.1',
        'sampling.arpc_temperature=0.1',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # Hub eval.py with use_block_cache=False: single-stream + small blocks +
    # truncated attention-block forwards (no DualCache replace path).
    # Hub ALWAYS confidence-unmasks (generation_functions.py threshold≈0.9–0.95);
    # omitting thr left this profile on mid-α ancestral — silent recipe bug.
    'hubmatch': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=0.9',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=8',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.ban_mask_pad_logits=false',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # Hub DualCache (use_block_cache=True). Paper C0 ~64% GSM used thr=1
    # confidence-until + greedy. Must pin thr — otherwise DualCache K/V runs
    # under mid-α ancestral commit (wrong product).
    'dual_cache': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=1.0',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=true',
        'sampling.single_stream_decode=true',
        # Hub eval.py small_block_size=8; paper §4 default sub-block.
        'sampling.sub_block_size=8',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.ban_mask_pad_logits=false',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
    ],
    # UCC — Hub DualCache *commit* analog on USDM (not DualCache K/V).
    # Commit = argmax(p_x0) + thr + force-max; undecided stays Unif; no Duo
    # scribble; revise OFF (Hub has none). sticky_min_conf=0 = Hub force-max.
    #
    # ``uniform_dual``: closest to masked ``dual_cache`` (thr=1, Hub ss+sub8).
    # ``uniform_commit``: bake-off UC twin of masked B1 (thr=0.9, dual pack).
    'uniform_dual': [
        'sampling.use_arpc=false',
        'sampling.uniform_confidence_sticky=true',
        'sampling.uniform_commit_revise=false',
        'sampling.sticky_min_conf=0.0',
        'sampling.unmask_threshold=1.0',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=8',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.ban_mask_pad_logits=true',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
    ],
    # Bake D5: same as ``uniform_dual`` but force-max site is random among
    # candidates (token = argmax still). Isolates confidence ranking.
    'uniform_dual_random': [
        'sampling.use_arpc=false',
        'sampling.uniform_confidence_sticky=true',
        'sampling.uniform_commit_revise=false',
        'sampling.uniform_commit_random=true',
        'sampling.uniform_commit_order=random',
        'sampling.sticky_min_conf=0.0',
        'sampling.unmask_threshold=1.0',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=8',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.ban_mask_pad_logits=true',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
    ],
    # Bake L→R: same as ``uniform_dual`` / ucc_thr1_sub8 but force-max is
    # leftmost unfrozen site (matched NFE if D2 is force-max dominated).
    'ucc_l2r_sub8': [
        'sampling.use_arpc=false',
        'sampling.uniform_confidence_sticky=true',
        'sampling.uniform_commit_revise=false',
        'sampling.uniform_commit_random=false',
        'sampling.uniform_commit_order=ltr',
        'sampling.sticky_min_conf=0.0',
        'sampling.unmask_threshold=1.0',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=8',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.ban_mask_pad_logits=true',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
    ],
    # Bake D4: D2 packing (ss+sub8) + quiet T, plain ancestral (no ARPC/UCC).
    # If this ≈ D2 GSM, UCC gain is mostly T/greedy/packing not sticky ranking.
    'ss_quiet_ancestral': [
        'sampling.use_arpc=false',
        'sampling.unmask_threshold=null',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=8',
        'sampling.greedy=false',
        'sampling.p_nucleus=1.0',
        'sampling.x0_temperature=0.1',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
        'sampling.uniform_confidence_sticky=false',
        'sampling.uniform_commit_revise=false',
        'sampling.uniform_commit_random=false',
    ],
    # Bake-off UC cell: Hub-clean commit, packing matched to masked B1.
    'uniform_commit': [
        'sampling.use_arpc=false',
        'sampling.uniform_confidence_sticky=true',
        'sampling.uniform_commit_revise=false',
        'sampling.sticky_min_conf=0.0',
        'sampling.unmask_threshold=0.9',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.ban_mask_pad_logits=true',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
    ],
    # DualCache schedule (thr=1) on B1 packing — isolates thr from ss pack.
    'uniform_commit_t1': [
        'sampling.use_arpc=false',
        'sampling.uniform_confidence_sticky=true',
        'sampling.uniform_commit_revise=false',
        'sampling.sticky_min_conf=0.0',
        'sampling.unmask_threshold=1.0',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=false',
        'sampling.sub_block_size=null',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.ban_mask_pad_logits=true',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
    ],
    # thr=0.9 + Hub ss packing (old unfair remask twin / hubmatch geometry).
    'uniform_commit_ss': [
        'sampling.use_arpc=false',
        'sampling.uniform_confidence_sticky=true',
        'sampling.uniform_commit_revise=false',
        'sampling.sticky_min_conf=0.0',
        'sampling.unmask_threshold=0.9',
        'sampling.hierarchical_kv=true',
        'sampling.use_block_cache=false',
        'sampling.single_stream_decode=true',
        'sampling.sub_block_size=8',
        'sampling.greedy=true',
        'sampling.p_nucleus=1.0',
        'sampling.ban_mask_pad_logits=true',
        'sampling.x0_temperature=1.0',
        'sampling.posterior_sampler=fast',
        'sampling.allow_full_seq_decode=false',
    ],
}

def _parse_hydra_override_value(raw: str) -> Any:
  raw = raw.strip()
  if raw in ('true', 'True'):
    return True
  if raw in ('false', 'False'):
    return False
  if raw in ('null', 'None'):
    return None
  try:
    return int(raw)
  except ValueError:
    pass
  try:
    return float(raw)
  except ValueError:
    return raw.strip('"').strip("'")


def _lm_eval_kwargs_from_overrides(overrides: list[str]) -> dict[str, Any]:
  """Build lm-eval kwargs from Hydra ``sampling.*=`` override tokens.

  Adds ``clear_unmask_threshold`` when ``unmask_threshold`` is present so
  ancestral profiles clear a sticky thr and confidence profiles keep it.
  Drops ``p_nucleus`` (lm-eval path does not pin it; default 1.0).
  """
  out: dict[str, Any] = {}
  for token in overrides:
    if '=' not in token:
      continue
    key, raw = token.split('=', 1)
    key = key.strip()
    if key.startswith('sampling.'):
      key = key[len('sampling.'):]
    if key == 'p_nucleus':
      continue
    out[key] = _parse_hydra_override_value(raw)
  if 'unmask_threshold' in out:
    out['clear_unmask_threshold'] = out['unmask_threshold'] is None
  return out


# Derived from DECODE_PROFILES — edit the Hydra table above, not this dict.
LM_EVAL_DECODE_PROFILES: dict[str, dict[str, Any]] = {
    name: _lm_eval_kwargs_from_overrides(overrides)
    for name, overrides in DECODE_PROFILES.items()
}

_CONVERSION_DATA_HINTS = (
    'nemotron', 'sft', 'instruct', 'chat', 'ultrachat', 'sharegpt',
)
_NATIVE_DATA_HINTS = (
    'openwebtext', 'owt', 'lm1b', 'fineweb',
)

META_SCHEMA_VERSION = 1


def infer_sample_mode(model_config) -> str:
  data = str(OmegaConf.select(model_config, 'data.train') or '').lower()
  tokenizer = str(
      OmegaConf.select(model_config, 'data.tokenizer_name_or_path') or '').lower()
  line = str(
      OmegaConf.select(model_config, 'line')
      or OmegaConf.select(model_config, 'experiment.line')
      or '').lower()
  if any(h in data for h in _CONVERSION_DATA_HINTS) or 'ar2block' in line:
    return 'conversion_free'
  if any(h in data for h in _NATIVE_DATA_HINTS) or line == 'block':
    return 'native_free'
  if 'instruct' in tokenizer:
    return 'conversion_free'
  return 'native_free'


def conversion_prefix_text(user_prompt: str | None = None) -> str:
  return _conversion_prefix_text(user_prompt)


def conversion_prefix_ids(tokenizer, user_prompt: str | None, device):
  import torch
  if user_prompt:
    text = apply_conversion_chat_template(
        tokenizer,
        [{'role': 'user', 'content': str(user_prompt)}],
        add_generation_prompt=True,
        tokenize=False,
    )
  else:
    text = conversion_prefix_text(None)
  ids = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids']
  return ids.to(device), text


def meta_path_for(samples_path: str | Path) -> Path:
  path = Path(samples_path)
  return path.with_suffix('.meta.json')


def write_samples_meta(samples_path: str | Path, meta: dict) -> Path:
  path = meta_path_for(samples_path)
  payload = dict(meta)
  payload.setdefault('schema_version', META_SCHEMA_VERSION)
  if 'code_fingerprint' not in payload:
    try:
      from discrete_diffusion.evaluations.code_fingerprint import (
          code_fingerprint_header,
      )
      payload['code_fingerprint'] = code_fingerprint_header()
    except Exception as e:  # noqa: BLE001
      payload['code_fingerprint'] = {'error': f'{type(e).__name__}: {e}'}
  path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
  return path


def read_samples_meta(samples_path: str | Path) -> dict | None:
  path = meta_path_for(samples_path)
  if not path.is_file():
    return None
  try:
    return json.loads(path.read_text(encoding='utf-8'))
  except Exception:
    return None


def samples_reusable(
    samples_path: str | Path,
    *,
    sample_mode: str,
    decode_profile: str,
    checkpoint_path: str | Path,
    force_regen: bool = False,
) -> bool:
  """True iff samples.pt + matching samples.meta.json exist for this request."""
  if force_regen:
    return False
  path = Path(samples_path)
  if not path.is_file():
    return False
  meta = read_samples_meta(path)
  if not meta:
    return False
  fp = infer_forward_from_checkpoint_path(checkpoint_path)
  profile = coerce_profile_for_forward(decode_profile, fp)
  # Never reuse illegal stored profiles (e.g. U0 baseline samples scored
  # as GenPPL after a later coerce-only generate path).
  stored = str(meta.get('decode_profile') or '').strip().lower()
  if fp in ('uniform', 'hybrid') and stored in (
      'baseline', 'dual_cache', 'full_seq_dual'):
    return False
  # Masked: refuse legacy full-seq dual meta for normal open-loop requests.
  if fp == 'masked' and stored == 'full_seq_dual' and profile != 'full_seq_dual':
    return False
  if stored != profile:
    return False
  try:
    ckpt_a = str(Path(checkpoint_path).expanduser().resolve())
    ckpt_b = str(Path(str(meta.get('checkpoint_path', ''))).expanduser().resolve())
  except Exception:
    return False
  return (
      str(meta.get('sample_mode')) == str(sample_mode)
      and stored == profile
      and ckpt_a == ckpt_b
  )


def infer_mode_from_checkpoint_path(checkpoint_path: str | Path) -> str | None:
  """Best-effort family mode from Hydra config beside a Lightning ckpt."""
  ckpt = Path(checkpoint_path).expanduser().resolve()
  candidates = (
      ckpt.parent.parent / 'hydra' / '.hydra' / 'config.yaml',
      ckpt.parent.parent / '.hydra' / 'config.yaml',
      ckpt.parent / '.hydra' / 'config.yaml',
      ckpt.parent.parent / 'config.yaml',
  )
  for candidate in candidates:
    if not candidate.is_file():
      continue
    try:
      return infer_sample_mode(OmegaConf.load(candidate))
    except Exception:
      continue
  return None


def should_reuse_samples(
    samples_path: str | Path,
    *,
    checkpoint_path: str | Path,
    sample_mode: str = 'auto',
    decode_profile: str = 'baseline',
    force_regen: bool = False,
) -> bool:
  """Eval-script gate: refuse reuse without matching ``samples.meta.json``.

  ``sample_mode=auto`` resolves via checkpoint Hydra config when possible;
  otherwise requires meta mode in ``{native_free, conversion_free}`` and a
  matching profile + checkpoint path. Uniform/hybrid coerce illegal profiles
  before the match (so ``baseline`` meta never scores as GenPPL).
  """
  if force_regen:
    return False
  mode = str(sample_mode or 'auto').strip().lower()
  fp = infer_forward_from_checkpoint_path(checkpoint_path)
  profile = coerce_profile_for_forward(decode_profile, fp)
  if mode == 'auto':
    resolved = infer_mode_from_checkpoint_path(checkpoint_path)
    if resolved is not None:
      return samples_reusable(
          samples_path,
          sample_mode=resolved,
          decode_profile=profile,
          checkpoint_path=checkpoint_path,
          force_regen=False,
      )
    path = Path(samples_path)
    if not path.is_file():
      return False
    meta = read_samples_meta(path)
    if not meta:
      return False
    if str(meta.get('sample_mode')) not in ('native_free', 'conversion_free'):
      return False
    stored = str(meta.get('decode_profile') or '').strip().lower()
    if fp in ('uniform', 'hybrid') and stored in ('baseline', 'dual_cache'):
      return False
    try:
      ckpt_a = str(Path(checkpoint_path).expanduser().resolve())
      ckpt_b = str(
          Path(str(meta.get('checkpoint_path', ''))).expanduser().resolve())
    except Exception:
      return False
    return stored == profile and ckpt_a == ckpt_b
  return samples_reusable(
      samples_path,
      sample_mode=mode,
      decode_profile=profile,
      checkpoint_path=checkpoint_path,
      force_regen=False,
  )


def infer_forward_from_checkpoint_path(
    checkpoint_path: str | Path,
) -> str | None:
  """Best-effort ``algo.forward_process_name`` from hydra cfg beside a ckpt."""
  ckpt = Path(checkpoint_path).expanduser()
  candidates = (
      ckpt.parent.parent / 'hydra' / '.hydra' / 'config.yaml',
      ckpt.parent.parent / '.hydra' / 'config.yaml',
      ckpt.parent / '.hydra' / 'config.yaml',
      ckpt.parent.parent / 'hydra' / 'config.yaml',
  )
  for cfg_path in candidates:
    if not cfg_path.is_file():
      continue
    try:
      cfg = OmegaConf.load(cfg_path)
      fp = OmegaConf.select(cfg, 'algo.forward_process_name')
      if fp:
        return str(fp).strip().lower()
    except Exception:
      continue
  return None


def allow_full_seq_decode_requested(cfg=None) -> bool:
  """True when operator explicitly opts into legacy full-seq dual open-loop."""
  import os
  env = os.environ.get('ALLOW_FULL_SEQ_DECODE', '').strip().lower()
  if env in ('1', 'true', 'yes'):
    return True
  if cfg is not None:
    raw = OmegaConf.select(cfg, 'sampling.allow_full_seq_decode')
    if raw is None and hasattr(cfg, 'get'):
      raw = cfg.get('allow_full_seq_decode', None)
    if raw is True or str(raw).strip().lower() in ('1', 'true', 'yes'):
      return True
  return False


def coerce_profile_for_forward(
    profile: str,
    forward_process_name: str | None,
    *,
    allow_full_seq: bool | None = None,
) -> str:
  """Normalize alias profiles; do **not** rewrite recipes across arms.

  * ``baseline`` → ``hierarchical`` (every forward process).
  * UCC profiles (``uniform_commit`` / ``uniform_dual`` / ``uniform_commit_t1``)
    stay as requested — they are the **UC** cell only, never a silent remap
    of ``hierarchical`` / ``hubmatch`` / ``dual_cache``.
  * DualCache K/V remains MASK-only; requesting ``dual_cache`` on uniform
    keeps the name (caller / bake must not pretend it is UCC).

  Full-seq dual refused unless ``allow_full_seq`` /
  ``ALLOW_FULL_SEQ_DECODE=1`` (legacy ``full_seq_dual``; C0 ancestral GSM
  2.27%% on that path).
  """
  name = str(profile or 'baseline').strip().lower() or 'baseline'
  if allow_full_seq is None:
    allow_full_seq = allow_full_seq_decode_requested()
  if allow_full_seq:
    if name == 'baseline':
      return 'full_seq_dual'
    return name
  if name == 'baseline':
    return 'hierarchical'
  return name


def profile_overrides(profile: str) -> list[str]:
  profile = str(profile or 'baseline').strip()
  if profile == 'keep':
    return []
  if profile not in DECODE_PROFILES:
    raise ValueError(
        f'decode_profile={profile!r} not in '
        f'{sorted(DECODE_PROFILES) + ["keep"]}')
  return list(DECODE_PROFILES[profile])


__all__ = [
    'DECODE_PROFILES',
    'LM_EVAL_DECODE_PROFILES',
    'META_SCHEMA_VERSION',
    'allow_full_seq_decode_requested',
    'coerce_profile_for_forward',
    'conversion_prefix_ids',
    'conversion_prefix_text',
    'infer_forward_from_checkpoint_path',
    'infer_mode_from_checkpoint_path',
    'infer_sample_mode',
    'meta_path_for',
    'profile_overrides',
    'read_samples_meta',
    'samples_reusable',
    'should_reuse_samples',
    'write_samples_meta',
]
