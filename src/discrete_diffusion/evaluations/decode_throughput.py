"""Decode throughput (tok/s) for ``BlockTrainer`` + ``BlockSampler``.

Measures wall-clock generation speed on our stack — honest UNI-D² decode,
not Fast-dLLM hierarchical KV / sub-block parallel unmask. Comparable only
when reported with the same hardware and ``num_steps`` / ``batch_size``.

Example::

  PYTHONPATH=src python -m discrete_diffusion.evaluations.decode_throughput \\
    checkpoint_path=outputs/block_qwen/ar2block_masked_139760/checkpoints/last.ckpt \\
    metrics_path=outputs/block_qwen/ar2block_masked_139760/lm_eval/tok_s.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from discrete_diffusion.data import get_tokenizer


def _load_model(checkpoint_path: str, device: torch.device):
  path = Path(hydra.utils.to_absolute_path(checkpoint_path))
  if not path.is_file():
    raise FileNotFoundError(path)
  ckpt = torch.load(path, map_location='cpu', weights_only=False)
  config = ckpt['hyper_parameters']['config']
  if not OmegaConf.is_config(config):
    config = OmegaConf.create(config)
  tokenizer = get_tokenizer(config)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  model = algo_cls.load_from_checkpoint(
      str(path), config=config, tokenizer=tokenizer, map_location=device)
  model.to(device)
  model.eval()
  return model, config, tokenizer


def _sync(device: torch.device) -> None:
  if device.type == 'cuda' and torch.cuda.is_available():
    torch.cuda.synchronize(device)


@torch.no_grad()
def _timed_generate(
    model,
    sampler,
    *,
    num_samples: int,
    num_steps: int,
    prefix_ids: torch.Tensor | None,
    device: torch.device,
    max_new_tokens: int | None,
) -> tuple[torch.Tensor, float]:
  _sync(device)
  t0 = time.perf_counter()
  samples = sampler.generate(
      model,
      num_samples=num_samples,
      num_steps=num_steps,
      eps=None,
      inject_bos=prefix_ids is None,
      prefix_ids=prefix_ids,
      max_new_tokens=max_new_tokens,
  )
  _sync(device)
  elapsed = time.perf_counter() - t0
  return samples, elapsed


def _count_new_tokens(
    samples: torch.Tensor,
    *,
    prefix_len: int,
    mask_id: int,
    eos_id: int | None,
    max_new_tokens: int | None = None,
) -> int:
  """Count continuation tokens (stop at EOS; ignore trailing masks)."""
  total = 0
  for row in samples:
    end = None if max_new_tokens is None else prefix_len + int(max_new_tokens)
    cont = row[prefix_len:end]
    if eos_id is not None:
      hits = (cont == eos_id).nonzero(as_tuple=False)
      if hits.numel() > 0:
        cont = cont[: int(hits[0]) + 1]
    # Drop pure mask pads if any remain.
    if mask_id is not None:
      keep = cont != mask_id
      total += int(keep.sum().item())
    else:
      total += int(cont.numel())
  return total


@hydra.main(
    config_path='../../../configs/eval',
    config_name='decode_throughput',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
  device = torch.device(
      cfg.device if torch.cuda.is_available() or cfg.device == 'cpu' else 'cpu')
  torch.set_grad_enabled(False)

  model, config, tokenizer = _load_model(cfg.checkpoint_path, device)
  sampler = model._create_sampler()
  if sampler is None:
    raise RuntimeError('no BlockSampler configured on checkpoint')

  num_steps = int(cfg.num_steps or getattr(config.sampling, 'steps', 32))
  batch_size = int(cfg.batch_size)
  num_batches = int(cfg.num_batches)
  warmup = int(cfg.warmup_batches)
  mode = str(cfg.mode)
  max_new_tokens = int(cfg.get('max_new_tokens', 0)) or None

  prefix_ids = None
  prefix_len = 0
  if mode == 'conditional':
    prompt = str(cfg.prompt)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')[
        'input_ids'].to(device)
    # Cap so there is room to generate.
    max_prefix = max(1, model.num_tokens - int(cfg.min_new_tokens))
    ids = ids[:, :max_prefix]
    prefix_len = int(ids.shape[1])
    if batch_size > 1:
      prefix_ids = ids.expand(batch_size, -1).contiguous()
    else:
      prefix_ids = ids

  # Warmup (not timed).
  for _ in range(warmup):
    _timed_generate(
        model, sampler,
        num_samples=batch_size,
        num_steps=num_steps,
        prefix_ids=prefix_ids,
        device=device,
        max_new_tokens=max_new_tokens,
    )

  elapsed_all = 0.0
  tokens_all = 0
  for _ in range(num_batches):
    samples, elapsed = _timed_generate(
        model, sampler,
        num_samples=batch_size,
        num_steps=num_steps,
        prefix_ids=prefix_ids,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    elapsed_all += elapsed
    tokens_all += _count_new_tokens(
        samples,
        prefix_len=prefix_len,
        mask_id=int(model.mask_id),
        eos_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )

  tok_s = tokens_all / max(elapsed_all, 1e-9)
  metrics = {
      'checkpoint_path': str(Path(cfg.checkpoint_path).resolve()),
      'device': str(device),
      'mode': mode,
      'batch_size': batch_size,
      'num_batches': num_batches,
      'warmup_batches': warmup,
      'num_steps': num_steps,
      'seq_len': int(model.num_tokens),
      'block_size': int(model.block_size),
      'prefix_len': prefix_len,
      'tokens_generated': tokens_all,
      'elapsed_s': elapsed_all,
      'tok_s': tok_s,
      'note': (
          'UNI-D2 BlockSampler throughput (no Fast-dLLM hierarchical KV / '
          'sub-block parallel unmask).'),
  }

  out = Path(hydra.utils.to_absolute_path(cfg.metrics_path))
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
  print(json.dumps(metrics, indent=2))
  print(f'Wrote {out}')


if __name__ == '__main__':
  main()
