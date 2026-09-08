"""Generate free / conditional samples from a BlockTrainer checkpoint.

Family-aware free sampling (matches native vs Instruct paper practice):

* ``native_free`` — LM free-gen (prior + BOS). Primary for ``LINE=block`` / OWT.
* ``conversion_free`` — open-ended gen from a train-aligned chat header
  (system + ``assistant`` turn), not bare ``<|im_start|>``. Primary free
  sample story for ``LINE=ar2block`` / Nemotron Instruct conversion.
* ``bare_bos`` — legacy bare-BOS ablation (often misleading on Instruct).

Default ``decode_profile=baseline`` clears confidence / ARPC / DualCache so
free-gen measures the shared ancestral ``BlockSampler``, not exactness overlays.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
import tqdm
from omegaconf import OmegaConf

from discrete_diffusion.evaluations.checkpoint_utils import (
    load_block_trainer_checkpoint,
)
from discrete_diffusion.evaluations.decode_profiles import (
    conversion_prefix_ids,
    infer_sample_mode,
    profile_overrides,
    write_samples_meta,
)


def _collect_sampling_overrides(cfg) -> list[str]:
  """Build ``sampling.*=`` override tokens for checkpoint merge.

  Order (last wins per key): nested ``sampling.*`` / legacy knobs first, then
  ``decode_profile`` clears (so baseline cannot be defeated by nested thr/ARPC),
  then explicit ``hydra_overrides`` (intentional CLI wins last).
  """
  overrides: list[str] = []

  sampling = cfg.get('sampling')
  if sampling is not None:
    flat = OmegaConf.to_container(sampling, resolve=True) or {}
    if isinstance(flat, dict):
      for key, val in flat.items():
        if val is None:
          continue
        if isinstance(val, bool):
          raw = 'true' if val else 'false'
        else:
          raw = str(val)
        overrides.append(f'sampling.{key}={raw}')

  for legacy in (
      'use_arpc', 'arpc_mode', 'arpc_corruption_mode', 'arpc_use_prefix_fill',
      'unmask_threshold', 'greedy', 'hierarchical_kv', 'use_block_cache',
      'single_stream_decode', 'sub_block_size', 'p_nucleus'):
    if legacy in cfg and cfg.get(legacy) is not None:
      val = cfg.get(legacy)
      if isinstance(val, bool):
        raw = 'true' if val else 'false'
      else:
        raw = str(val)
      overrides.append(f'sampling.{legacy}={raw}')

  greedy = cfg.get('greedy', None)
  if greedy is not None and str(greedy).strip().lower() not in ('', 'null', 'none'):
    overrides.append(
        f"sampling.greedy={'true' if bool(greedy) else 'false'}")

  profile = str(cfg.get('decode_profile', 'baseline') or 'baseline').strip()
  overrides.extend(profile_overrides(profile))

  extra = cfg.get('hydra_overrides') or []
  if isinstance(extra, str):
    extra = [extra]
  for token in list(extra):
    token = str(token).strip()
    if token:
      overrides.append(token)

  seen: dict[str, str] = {}
  for token in overrides:
    if '=' not in token:
      continue
    key = token.split('=', 1)[0]
    seen[key] = token
  return list(seen.values())


def _truncate_im_end(text: str) -> str:
  stop = '<|im_end|>'
  if stop in text:
    return text.split(stop, 1)[0] + stop
  return text


@hydra.main(
    config_path='../../../configs/eval',
    config_name='generate_samples',
    version_base='1.3')
def main(cfg):
  device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
  torch.set_float32_matmul_precision('high')
  torch.set_grad_enabled(False)

  print(f'Loading checkpoint from {cfg.checkpoint_path}')
  checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint_path)
  if not Path(checkpoint_path).exists():
    raise FileNotFoundError(f'Checkpoint not found at {checkpoint_path}')

  overrides = _collect_sampling_overrides(cfg)
  if overrides:
    print('Decode overrides:', overrides)

  print('Loading tokenizer and model...')
  model, model_config, tokenizer = load_block_trainer_checkpoint(
      checkpoint_path, device, hydra_overrides=overrides or None)
  print(f'Detected algorithm class: {model.__class__.__name__}')

  mode = str(cfg.get('sample_mode', 'auto') or 'auto').strip().lower()
  if mode == 'auto':
    mode = infer_sample_mode(model_config)
  if mode not in ('native_free', 'conversion_free', 'bare_bos'):
    raise ValueError(
        f'sample_mode={mode!r} not in '
        f'(auto|native_free|conversion_free|bare_bos)')

  prefix_ids = None
  inject_bos = True
  prefix_text = None
  user_prompt = cfg.get('conversion_user_prompt', None)
  if user_prompt is not None and str(user_prompt).strip().lower() in (
      '', 'null', 'none'):
    user_prompt = None

  if mode == 'conversion_free':
    prefix_ids, prefix_text = conversion_prefix_ids(
        tokenizer, user_prompt, device)
    inject_bos = False
  elif mode == 'bare_bos':
    inject_bos = True
    prefix_ids = None
  else:  # native_free
    raw_inj = OmegaConf.select(model_config, 'sampling.inject_bos')
    inject_bos = True if raw_inj is None else bool(raw_inj)
    prefix_ids = None

  print(
      f'Sample mode={mode} inject_bos={inject_bos} '
      f'prefix_len={0 if prefix_ids is None else int(prefix_ids.shape[-1])} '
      f'decode_profile={cfg.get("decode_profile")} '
      f'use_arpc={OmegaConf.select(model_config, "sampling.use_arpc")} '
      f'unmask_threshold={OmegaConf.select(model_config, "sampling.unmask_threshold")}'
  )
  if prefix_text:
    print('Prefix text:', repr(prefix_text[:180]))

  if cfg.torch_compile:
    print('Compiling model...')
    model = torch.compile(model)

  sampler = model._create_sampler()
  if sampler is None:
    raise RuntimeError('Checkpoint has no BlockSampler')

  num_samples = int(cfg.num_samples)
  batch_size = int(cfg.batch_size)
  num_steps = cfg.num_steps
  max_new = cfg.get('max_new_tokens', None)
  if max_new is not None and str(max_new).strip().lower() in ('', 'null', 'none'):
    max_new = None
  elif max_new is not None:
    max_new = int(max_new)

  print(
      f'Generating {num_samples} samples '
      f'(batch_size={batch_size}, steps={num_steps or "default"}, '
      f'max_new_tokens={max_new})')

  all_samples = []
  with tqdm.tqdm(total=num_samples, desc='Sampling', dynamic_ncols=True) as pbar:
    for i in range(0, num_samples, batch_size):
      current_batch_size = min(batch_size, num_samples - i)
      pref = None
      if prefix_ids is not None:
        pref = prefix_ids.expand(current_batch_size, -1).contiguous()
      samples = sampler.generate(
          model,
          num_samples=current_batch_size,
          num_steps=num_steps,
          eps=None,
          inject_bos=inject_bos,
          prefix_ids=pref,
          max_new_tokens=max_new,
      )
      all_samples.append(samples.detach().cpu())
      pbar.update(current_batch_size)

  all_samples = torch.cat(all_samples, dim=0)
  out_path = Path(hydra.utils.to_absolute_path(cfg.samples_path))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(all_samples, out_path)
  print(f'Saved {len(all_samples)} samples to {out_path}')

  meta = {
      'checkpoint_path': str(Path(checkpoint_path).resolve()),
      'sample_mode': mode,
      'decode_profile': str(cfg.get('decode_profile')),
      'inject_bos': inject_bos,
      'prefix_text': prefix_text,
      'prefix_len': 0 if prefix_ids is None else int(prefix_ids.shape[-1]),
      'max_new_tokens': max_new,
      'num_samples': int(all_samples.shape[0]),
      'seq_len': int(all_samples.shape[1]),
      'note': (
          'conversion_free = Instruct-aligned open assistant header; '
          'native_free = LM free-gen; bare_bos = legacy ablation.'),
  }
  meta_path = write_samples_meta(out_path, meta)
  print(f'Wrote {meta_path}')

  if cfg.get('save_text', False):
    print('Decoding samples to text...')
    from discrete_diffusion.data.tokenizers import Text8Tokenizer
    stop = bool(cfg.get('stop_at_im_end', True))
    if isinstance(tokenizer, Text8Tokenizer):
      texts = [
          tokenizer.decode_ids_to_text(s, skip_special_tokens=False)
          for s in all_samples]
    else:
      texts = tokenizer.batch_decode(all_samples, skip_special_tokens=False)
    if stop:
      texts = [_truncate_im_end(t) for t in texts]
    text_path = out_path.with_suffix('.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
      f.write(f'# sample_mode={mode} decode_profile={cfg.get("decode_profile")}\n')
      if prefix_text:
        f.write(f'# prefix={prefix_text!r}\n')
      for i, text in enumerate(texts):
        f.write(f'Sample {i}:\n{text}\n{"-" * 80}\n')
    print(f'Saved text samples to {text_path}')


if __name__ == '__main__':
  main()
