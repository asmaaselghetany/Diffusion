"""Standalone Generative Perplexity evaluation.

Loads generated samples and evaluates NLL/PPL with a chosen eval LM.

Supported sample formats:
- .pt: torch.Tensor of token ids (shape [N, T] or [N, 1, T])
- .npz: numpy array under key 'samples' (shape [N, T])
- .json: contains base64-encoded numpy array under key 'np_tokens_b64'

This script decodes using `model_tokenizer` and (optionally) retokenizes with
the eval model's tokenizer before computing loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.tokenizers import Text8Tokenizer
from .decode_profiles import read_samples_meta


def _trim_token_rows_at_eos(
    rows: np.ndarray,
    eos_id: int | None,
    *,
    prefix_len: int = 0,
    pad_id: int | None = None,
) -> np.ndarray:
  """Trim at first EOS at or after ``prefix_len`` (skip chat-header im_end).

  Right-pad shorter rows with ``pad_id`` (default ``eos_id``, never bare 0 —
  Qwen id 0 is ``!`` and was polluting first_chunk GenPPL text).
  """
  if eos_id is None:
    return rows
  prefix_len = max(0, int(prefix_len))
  fill = int(eos_id if pad_id is None else pad_id)
  trimmed = []
  max_len = 0
  for row in rows:
    search = row[prefix_len:]
    hits = np.where(search == eos_id)[0]
    if hits.size:
      row = row[: prefix_len + int(hits[0]) + 1]
    trimmed.append(row)
    max_len = max(max_len, row.shape[0])
  out = np.full((len(trimmed), max_len), fill, dtype=rows.dtype)
  for i, row in enumerate(trimmed):
    out[i, : row.shape[0]] = row
  return out


def _eos_token_stats(
    rows: np.ndarray,
    eos_id: int | None,
    *,
    prefix_len: int = 0,
) -> dict:
  """Honesty metrics: how often / how early EOS appears *after* the prefix.

  Chat ``conversion_free`` prefixes embed ``<|im_end|>`` (Qwen eos) after the
  system turn. Counting that as generation EOS falsely reports collapse and
  flattens first_chunk_only PPL.
  """
  n, t = int(rows.shape[0]), int(rows.shape[1])
  prefix_len = max(0, int(prefix_len))
  if eos_id is None:
    return {
        'eos_rate': None,
        'mean_tokens_before_eos': float(t),
        'median_tokens_before_eos': float(t),
        'full_length_rate': 1.0,
        'raw_seq_len': t,
        'prefix_len': prefix_len,
        'num_samples': n,
    }
  lengths = []
  has_eos = 0
  for row in rows:
    search = row[prefix_len:]
    hits = np.where(search == eos_id)[0]
    if hits.size:
      has_eos += 1
      lengths.append(prefix_len + int(hits[0]) + 1)
    else:
      lengths.append(t)
  arr = np.asarray(lengths, dtype=np.float64)
  return {
      'eos_rate': float(has_eos / max(n, 1)),
      'mean_tokens_before_eos': float(arr.mean()) if n else 0.0,
      'median_tokens_before_eos': float(np.median(arr)) if n else 0.0,
      'full_length_rate': float(np.mean(arr >= t)) if n else 0.0,
      'raw_seq_len': t,
      'prefix_len': prefix_len,
      'num_samples': n,
  }


def _unigram_shannon_entropy(token_ids: np.ndarray | list[int]) -> float:
  """Shannon entropy (nats) of the empirical unigram over ``token_ids``.

  Unifusion-style hygiene: low GenPPL + low H → repetitive collapse; high H alone
  can be noise. Empty / single-token → 0.
  """
  ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
  if ids.size == 0:
    return 0.0
  _, counts = np.unique(ids, return_counts=True)
  p = counts.astype(np.float64) / float(ids.size)
  return float(-(p * np.log(p)).sum())


def _strip_conversion_prefix(
    body: str,
    prefix_text: str | None,
    *,
    model_tokenizer=None,
) -> str:
  """Strip chat header whether or not specials were kept in ``body``."""
  if not prefix_text:
    return body
  if body.startswith(prefix_text):
    return body[len(prefix_text):]
  if model_tokenizer is None:
    return body
  # Body usually comes from skip_special_tokens=True decode; prefix_text keeps
  # ``<|im_start|>`` etc. Match the specials-stripped prefix instead.
  try:
    pref_ids = model_tokenizer.encode(prefix_text, add_special_tokens=False)
    stripped = model_tokenizer.decode(pref_ids, skip_special_tokens=True)
  except Exception:
    return body
  if stripped and body.startswith(stripped):
    return body[len(stripped):]
  return body


def _unigram_entropy_stats(
    texts: List[str],
    tokenizer: AutoTokenizer,
    *,
    prefix_text: str | None = None,
    model_tokenizer=None,
) -> dict:
  """Mean sample-level unigram entropy (Unifusion C.3 protocol, single seed).

  Each text is retokenized with ``tokenizer`` (``add_special_tokens=False``).
  If ``prefix_text`` is set, it is stripped once from the start so chat headers
  do not dominate H.
  """
  hs: List[float] = []
  lengths: List[int] = []
  for text in texts:
    body = _strip_conversion_prefix(
        text, prefix_text, model_tokenizer=model_tokenizer)
    body = body.strip()
    if not body:
      hs.append(0.0)
      lengths.append(0)
      continue
    ids = tokenizer.encode(body, add_special_tokens=False)
    hs.append(_unigram_shannon_entropy(ids))
    lengths.append(len(ids))
  arr = np.asarray(hs, dtype=np.float64)
  len_arr = np.asarray(lengths, dtype=np.float64)
  return {
      'unigram_entropy_mean': float(arr.mean()) if arr.size else 0.0,
      'unigram_entropy_median': float(np.median(arr)) if arr.size else 0.0,
      'unigram_entropy_std': float(arr.std(ddof=0)) if arr.size else 0.0,
      'unigram_entropy_tokenizer': str(
          getattr(tokenizer, 'name_or_path', type(tokenizer).__name__)),
      'unigram_entropy_mean_tokens': float(len_arr.mean()) if len_arr.size else 0.0,
      'unigram_entropy_num_samples': int(arr.size),
      'unigram_entropy_note': (
          'Shannon H (nats) of empirical unigram per sample, then mean; '
          'strip conversion_free prefix when present (Unifusion-style hygiene)'),
  }


def _decode_samples(model_tokenizer, z_ts: np.ndarray) -> List[str]:
  if isinstance(model_tokenizer, Text8Tokenizer):
    return [
        model_tokenizer.decode_ids_to_text(row, skip_special_tokens=True)
        for row in z_ts]
  return model_tokenizer.batch_decode(z_ts, skip_special_tokens=True)


def _load_samples(samples_path: str) -> np.ndarray:
  path = Path(hydra.utils.to_absolute_path(samples_path))
  if not path.exists():
    raise FileNotFoundError(f"Samples not found at {path}")

  if path.suffix == ".pt":
    z_ts = torch.load(path, weights_only=True)
    if isinstance(z_ts, torch.Tensor):
      arr = z_ts.detach().cpu()
    else:
      # if saved as dict or list, try common keys/shapes
      raise ValueError("Unsupported .pt structure; expected a Tensor.")
    if arr.ndim == 3 and arr.shape[1] == 1:
      arr = arr.squeeze(1)
    return arr.numpy()

  if path.suffix == ".npz":
    content = np.load(path)
    if 'samples' not in content:
      raise KeyError(".npz must contain 'samples' key")
    return content['samples']

  if path.suffix == ".json":
    from ..utils import utils as _utils
    with open(path, 'r') as f:
      payload = json.load(f)
    if 'np_tokens_b64' not in payload:
      raise KeyError(".json must contain 'np_tokens_b64' key")
    arr = _utils.base64_to_np(payload['np_tokens_b64'])
    return arr

  raise ValueError(f"Unsupported samples format: {path.suffix}")


def _retokenize(
  texts: List[str],
  tokenizer: AutoTokenizer,
  max_length: int,
  device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
  # Default context windows for common models; conservative fallback
  eval_context_size = 4096 if 'llama' in tokenizer.name_or_path.lower() else 1024
  batch = tokenizer(
    texts,
    return_tensors="pt",
    return_token_type_ids=False,
    return_attention_mask=True,
    truncation=True,
    padding=True,
    max_length=max_length,
  )
  attn_mask = batch['attention_mask'].to(device)
  input_ids = batch['input_ids'].to(device)
  return input_ids, attn_mask, eval_context_size


@hydra.main(config_path='../../../configs/eval', config_name='gen_ppl', version_base='1.3')
def main(cfg):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_float32_matmul_precision('high')
  torch.set_grad_enabled(False)

  samples_path = Path(hydra.utils.to_absolute_path(cfg.samples_path))
  meta = read_samples_meta(samples_path)
  require_meta = bool(cfg.get('require_samples_meta', False))
  if require_meta and not meta:
    raise SystemExit(
        f'REFUSED: gen-PPL headline requires samples.meta.json beside '
        f'{samples_path} (regenerate via eval.sh / generate_samples). '
        f'Set require_samples_meta=false only for legacy ablations.')

  # Decode tokens (from diffusion model) to text using its tokenizer
  from discrete_diffusion.data import load_tokenizer_by_name
  model_tokenizer = load_tokenizer_by_name(str(cfg.model_tokenizer))

  eval_model = AutoModelForCausalLM.from_pretrained(cfg.pretrained_model, device_map="auto")
  eval_tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
  if eval_tokenizer.pad_token_id is None:
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

  if cfg.torch_compile:
    eval_model = torch.compile(eval_model)

  # Load samples and make text
  z_ts = _load_samples(cfg.samples_path)
  if z_ts.ndim != 2:
    raise ValueError(f"Expected 2D [N, T] tokens array, got {z_ts.shape}")
  prefix_len = 0
  if meta:
    try:
      prefix_len = int(meta.get('prefix_len') or 0)
    except (TypeError, ValueError):
      prefix_len = 0
  eos_stats = _eos_token_stats(
      z_ts, model_tokenizer.eos_token_id, prefix_len=prefix_len)
  if cfg.first_chunk_only:
    z_ts = _trim_token_rows_at_eos(
        z_ts, model_tokenizer.eos_token_id, prefix_len=prefix_len)
  texts = _decode_samples(model_tokenizer, z_ts)
  nonempty = sum(1 for t in texts if t.strip())
  prefix_text = None
  if meta:
    raw_pref = meta.get('prefix_text')
    if raw_pref is not None and str(raw_pref).strip():
      prefix_text = str(raw_pref)
  # Entropy needs only the eval tokenizer (cheap); compute even if GenPPL later fails.
  entropy_stats = _unigram_entropy_stats(
      texts, eval_tokenizer, prefix_text=prefix_text,
      model_tokenizer=model_tokenizer)

  if nonempty == 0:
    print('WARNING: all decoded samples are empty; writing null gen-PPL metrics.')
    metrics = {
        "file": Path(cfg.samples_path).stem,
        "pretrained_model": cfg.pretrained_model,
        "model_tokenizer": str(cfg.model_tokenizer),
        "warning": "all_decoded_samples_empty",
        "num_samples": int(len(texts)),
        "median_nll": None,
        "avg_nll": None,
        "ppl": None,
        "acc": None,
        "tokens": 0,
        "retokenize": bool(cfg.retokenize),
        "first_chunk_only": bool(cfg.first_chunk_only),
        "require_samples_meta": require_meta,
        "samples_meta": meta,
        **eos_stats,
        **entropy_stats,
    }
    print(json.dumps(metrics, indent=2))
    out_path = Path(hydra.utils.to_absolute_path(cfg.metrics_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
      json.dump(metrics, f)
    print(f"Saved metrics to: {out_path}")
    return

  total_acc = 0.0
  total_nll = 0.0
  total_tokens = 0.0
  all_nlls: List[float] = []

  with torch.no_grad():
    for i in range(0, len(texts), cfg.batch_size):
      xs = [t for t in texts[i:i + cfg.batch_size] if t.strip()]
      if not xs:
        continue

      if cfg.retokenize:
        input_ids, attn_mask, context_size = _retokenize(
          xs, eval_tokenizer, cfg.max_length, device)
      else:
        # Use model tokens directly (not recommended across tokenizers)
        # Here we re-tokenize anyway but with the same tokenizer to ensure tensors
        input_ids, attn_mask, context_size = _retokenize(
          xs, eval_tokenizer, cfg.max_length, device)

      # Evaluate possibly only the first chunk up to EOS
      logits = eval_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits[:, :-1]
      labels = input_ids[:, 1:]
      loss_mask = attn_mask[:, :-1]

      nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)

      if cfg.first_chunk_only:
        eos_id = eval_tokenizer.eos_token_id
        eos_mask = (labels == eos_id).cumsum(-1) == 0  # valid until first EOS (exclusive)
        # Ensure we still respect attention mask
        valid = loss_mask.bool() & eos_mask
      else:
        valid = loss_mask.bool()

      valid = valid.to(nll.dtype)
      all_nlls.extend(nll[valid == 1].detach().cpu().numpy().tolist())
      total_nll += float((nll * valid).sum().item())

      acc = (logits.argmax(-1) == labels).to(nll.dtype)
      total_acc += float((acc * valid).sum().item())
      total_tokens += float(valid.sum().item())

  if total_tokens == 0:
    raise RuntimeError("No valid tokens for evaluation (check inputs/EOS handling)")

  avg_nll = total_nll / total_tokens
  ppl = float(np.exp(avg_nll))
  acc = total_acc / total_tokens

  metrics = {
    "file": Path(cfg.samples_path).stem,
    "pretrained_model": cfg.pretrained_model,
    "median_nll": float(np.median(all_nlls)) if all_nlls else float('nan'),
    "avg_nll": float(avg_nll),
    "ppl": float(ppl),
    "acc": float(acc),
    "tokens": int(total_tokens),
    "retokenize": bool(cfg.retokenize),
    "first_chunk_only": bool(cfg.first_chunk_only),
    "require_samples_meta": require_meta,
    "samples_meta": meta,
    **eos_stats,
    **entropy_stats,
  }
  if bool(cfg.first_chunk_only) and eos_stats.get('eos_rate') is not None:
    # Flag flattering short-chunk PPL after early *post-prefix* collapse.
    gen_span = float(eos_stats['mean_tokens_before_eos']) - float(
        eos_stats.get('prefix_len') or 0)
    if float(eos_stats['eos_rate']) > 0.5 and gen_span < 64:
      metrics['honesty_warning'] = (
          'high_eos_rate_short_span: first_chunk_only PPL may launder collapse')
  # Unifusion-style: low GenPPL with low unigram H is often repetitive collapse.
  h_mean = float(entropy_stats.get('unigram_entropy_mean') or 0.0)
  if float(ppl) < 80.0 and h_mean < 4.5:
    prev = metrics.get('honesty_warning')
    extra = 'low_ppl_low_unigram_entropy: possible repetitive collapse'
    metrics['honesty_warning'] = f'{prev}; {extra}' if prev else extra
  # Quiet/T≪1 digit-soup: GPT-2 PPL can look excellent (~5) while samples are
  # long runs of digits / multilingual junk that almost never EOS. Unigram H
  # stays moderate (~6) so the low-H gate above misses it.
  flr = eos_stats.get('full_length_rate')
  if flr is not None and float(ppl) < 20.0 and float(flr) > 0.75:
    prev = metrics.get('honesty_warning')
    extra = (
        'low_ppl_high_full_length: likely digit/repetition soup '
        '(do not cite as fluency — inspect samples.txt)')
    metrics['honesty_warning'] = f'{prev}; {extra}' if prev else extra
  # High full-length rate = rarely stops; GenPPL≠linguistic fluency (not a
  # collapse gate — separate note so hygiene pairs stay citeable when PPL
  # is ordinary).
  if flr is not None and float(flr) > 0.75:
    metrics['fluency_note'] = (
        'high_full_length_rate: GenPPL+H is collapse hygiene only, '
        'not linguistic fluency (models rarely emit EOS)')
  # Prefix-EOS / empty-span scores must not be cited.
  n_samp = int(metrics.get('num_samples') or eos_stats.get('num_samples') or 0)
  if n_samp > 0 and int(total_tokens) < max(32, n_samp * 32):
    prev = metrics.get('honesty_warning')
    extra = (
        f'too_few_scored_tokens ({int(total_tokens)} for n={n_samp}): '
        'likely prefix-EOS bug or empty gens — do not cite GenPPL')
    metrics['honesty_warning'] = f'{prev}; {extra}' if prev else extra
    metrics['citeable'] = False
  else:
    w = str(metrics.get('honesty_warning') or '')
    metrics['citeable'] = not any(
        tag in w for tag in (
            'too_few_scored_tokens',
            'high_eos_rate_short_span',
            'low_ppl_high_full_length',
            'low_ppl_low_unigram_entropy',
        ))

  print(json.dumps(metrics, indent=2))
  out_path = Path(hydra.utils.to_absolute_path(cfg.metrics_path))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, 'w') as f:
    json.dump(metrics, f)
  print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
  main()
