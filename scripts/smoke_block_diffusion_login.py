#!/usr/bin/env python3
"""Login-node smoke checks for BlockDiffusion before submitting SLURM jobs."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional


@dataclass
class CheckResult:
  name: str
  ok: bool
  detail: str = ""


def _ok(name: str, detail: str = "") -> CheckResult:
  return CheckResult(name, True, detail)


def _fail(name: str, detail: str) -> CheckResult:
  return CheckResult(name, False, detail)


def check_venv(repo_root: Path) -> CheckResult:
  py = repo_root / ".venv" / "bin" / "python"
  if not py.is_file():
    return _fail("venv", f"Missing {py} — run setup_cuda_venv.sh")
  return _ok("venv", str(py))


def check_torch(repo_root: Path) -> CheckResult:
  py = str(repo_root / ".venv" / "bin" / "python")
  import subprocess

  code = (
    "import torch; "
    "print(f'torch={torch.__version__} cuda_built={torch.version.cuda} "
    f"cuda_available={{torch.cuda.is_available()}}')"
  )
  out = subprocess.check_output([py, "-c", code], text=True).strip()
  if "+cu" not in out and "cuda_built=None" in out:
    return _fail("torch", f"CPU-only build: {out}")
  return _ok("torch", out)


def check_triton_flex(repo_root: Path) -> CheckResult:
  try:
    from discrete_diffusion.compat.triton_shim import ensure_triton_attrs_descriptor
    ensure_triton_attrs_descriptor()
    import triton  # noqa: F401
    from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
    from discrete_diffusion.models.block_dit import BlockDiT, FLEX_ATTN_AVAILABLE, get_flex_runtime  # noqa: F401
  except Exception as exc:
    return _fail("triton_flex", f"{exc}")
  detail = f"triton={triton.__version__} flex_attn={FLEX_ATTN_AVAILABLE} runtime={get_flex_runtime()}"
  return _ok("triton_flex", detail)


def check_train_imports() -> CheckResult:
  try:
    from discrete_diffusion.training.pretrained import (  # noqa: F401
      load_matching_weights,
      resolve_pretrained_source,
    )
    from discrete_diffusion.train import train  # noqa: F401
  except Exception as exc:
    return _fail("train_imports", f"{exc}")
  return _ok("train_imports")


def check_compute_sync(compute_root: Path) -> CheckResult:
  missing = []
  for rel in (".venv/bin/python", "src/discrete_diffusion/__main__.py"):
    if not (compute_root / rel).exists():
      missing.append(rel)
  if missing:
    return _fail(
      "compute_sync",
      f"{compute_root} missing {', '.join(missing)} — run setup_compute_workspace.sh",
    )
  return _ok("compute_sync", str(compute_root))


def check_data_cache(
    cache_dir: Path,
    *,
    scratch: bool,
    pretrain: bool,
) -> CheckResult:
  required = []
  if scratch:
    required.append("openwebtext-train_train_bs1024_wrapped.dat")
  if pretrain:
    required.append("openwebtext-train_train_bs1024_wrapped_eosFalse_specialFalse.dat")
  missing = [name for name in required if not (cache_dir / name).exists()]
  if missing:
    hint = "PRETRAIN_DATA=1 bash slurm_scripts/block_diffusion/prep_data.sh" if pretrain else (
      "bash slurm_scripts/block_diffusion/prep_data.sh"
    )
    return _fail("data_cache", f"{cache_dir} missing {missing} — {hint}")
  return _ok("data_cache", str(cache_dir))


def check_checkpoint(cache_dir: Path, ckpt_name: str, download_hint: str) -> CheckResult:
  path = cache_dir / "checkpoints" / ckpt_name
  if not path.is_file():
    return _fail("checkpoint", f"Missing {path} — {download_hint}")
  return _ok("checkpoint", str(path))


def check_pretrained_load(repo_root: Path, ckpt: Path) -> CheckResult:
  from hydra import compose, initialize_config_dir
  from hydra.core.global_hydra import GlobalHydra

  from discrete_diffusion.data import get_tokenizer
  from discrete_diffusion.training.pretrained import load_matching_weights

  GlobalHydra.instance().clear()
  config_dir = str(repo_root / "configs")
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(
      config_name="config",
      overrides=[
        "model=block_dit",
        "algo=block_diffusion_masked",
        "data=tiny_shakespeare",
        f"data.cache_dir={repo_root / 'data_cache'}",
        "model.length=128",
        "block_size=16",
        "model.adaln=True",
        "algo.cross_attn=False",
        "model.attn_backend=sdpa",
        f"training.from_pretrained={ckpt}",
      ],
    )
  tok = get_tokenizer(cfg)
  import torch
  from discrete_diffusion.algorithms.block_diffusion import BlockDiffusion

  model = BlockDiffusion(cfg, tok)
  loaded, total, _skipped = load_matching_weights(model, str(ckpt))
  return _ok("pretrained_load", f"loaded={loaded}/{total}")


def check_forward_pass(
    repo_root: Path,
    *,
    attn_backend: str,
    cross_attn: bool,
    adaln: bool,
    seq_len: int = 128,
    use_cuda: bool = False,
) -> CheckResult:
  import torch
  from hydra import compose, initialize_config_dir
  from hydra.core.global_hydra import GlobalHydra

  from discrete_diffusion.algorithms.block_diffusion import BlockDiffusion
  from discrete_diffusion.data import get_tokenizer

  _register_hydra_resolvers()
  GlobalHydra.instance().clear()
  config_dir = str(repo_root / "configs")
  overrides = [
    "model=block_dit",
    "algo=block_diffusion_masked",
    "data=tiny_shakespeare",
    f"data.cache_dir={repo_root / 'data_cache'}",
    f"model.length={seq_len}",
    "block_size=16",
    f"model.adaln={'True' if adaln else 'False'}",
    f"algo.cross_attn={'True' if cross_attn else 'False'}",
    f"model.attn_backend={attn_backend}",
    "loader.eval_batch_size=2",
    "loader.batch_size=2",
  ]
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="config", overrides=overrides)

  tok = get_tokenizer(cfg)
  model = BlockDiffusion(cfg, tok)
  device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
  model = model.to(device)
  model.eval()

  seq = seq_len * 2 if cross_attn else seq_len
  x = torch.randint(0, min(1000, tok.vocab_size), (2, seq), device=device)
  t = torch.rand(2, device=device)
  if attn_backend == "flex" and device.type == "cpu":
    return _ok(
      "forward_pass",
      "flex model built; forward skipped on CPU — run smoke_test_gpu.sh",
    )
  try:
    with torch.no_grad():
      out = model.backbone(x, t)
  except Exception as exc:
    if attn_backend == "flex":
      return _fail(
        "forward_pass",
        f"flex forward failed: {exc}. Run smoke_test_gpu.sh on Booster to validate Triton kernels.",
      )
    raise
  shape = tuple(out.shape) if hasattr(out, "shape") else type(out).__name__
  from discrete_diffusion.models.block_dit import get_flex_runtime
  runtime = get_flex_runtime() if attn_backend == "flex" else attn_backend
  return _ok("forward_pass", f"backend={attn_backend} runtime={runtime} device={device} out={shape}")


def _register_hydra_resolvers() -> None:
  import functools
  import operator

  import omegaconf
  import torch

  def register(name: str, resolver) -> None:
    if not omegaconf.OmegaConf.has_resolver(name):
      omegaconf.OmegaConf.register_new_resolver(name, resolver)

  register("cwd", os.getcwd)
  register("device_count", torch.cuda.device_count)
  register("div_up", lambda x, y: (x + y - 1) // y)
  register(
    "mul",
    lambda *args: functools.reduce(operator.mul, args) if args else ValueError("mul"),
  )
  register("sub", lambda x, y: x - y)


def check_hydra_compose(repo_root: Path, overrides: List[str]) -> CheckResult:
  from hydra import compose, initialize_config_dir
  from hydra.core.global_hydra import GlobalHydra

  _register_hydra_resolvers()
  GlobalHydra.instance().clear()
  config_dir = str(repo_root / "configs")
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="config", overrides=overrides)
  return _ok("hydra_compose", f"algo={cfg.algo._target_.split('.')[-1]} data={cfg.data.train}")


def check_dataloader_step(repo_root: Path, data: str, cache_dir: Path) -> CheckResult:
  from hydra import compose, initialize_config_dir
  from hydra.core.global_hydra import GlobalHydra

  from discrete_diffusion.data import get_dataloaders, get_tokenizer

  _register_hydra_resolvers()
  GlobalHydra.instance().clear()
  config_dir = str(repo_root / "configs")
  with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(
      config_name="config",
      overrides=[
        f"data={data}",
        f"data.cache_dir={cache_dir}",
        "model.length=1024",
        "loader.global_batch_size=2",
        "loader.batch_size=2",
        "loader.eval_batch_size=2",
        "loader.num_workers=2",
        "+loader.persistent_workers=false",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.accelerator=cpu",
      ],
    )
  tok = get_tokenizer(cfg)
  train_loader, _ = get_dataloaders(cfg, tok)
  batch = next(iter(train_loader))
  if isinstance(batch, dict):
    batch = batch.get("input_ids", next(iter(batch.values())))
  if isinstance(batch, (list, tuple)):
    batch = batch[0]
  shape = tuple(batch.shape) if hasattr(batch, "shape") else type(batch).__name__
  return _ok("dataloader", f"data={data} batch_shape={shape}")


def run_checks(checks: Iterable[Callable[[], CheckResult]]) -> List[CheckResult]:
  results: List[CheckResult] = []
  for fn in checks:
    try:
      results.append(fn())
    except Exception:
      results.append(_fail(fn.__name__.replace("check_", ""), traceback.format_exc().strip()))
  return results


def print_results(results: List[CheckResult]) -> int:
  width = max(len(r.name) for r in results) if results else 10
  failed = 0
  for r in results:
    status = "PASS" if r.ok else "FAIL"
    line = f"[{status}] {r.name.ljust(width)}"
    if r.detail:
      line += f"  {r.detail}"
    print(line)
    if not r.ok:
      failed += 1
  print("-" * 60)
  if failed:
    print(f"{failed} check(s) failed — fix before sbatch")
    return 1
  print("All checks passed — safe to submit SLURM jobs (after setup_compute_workspace.sh)")
  return 0


def profile_tiny(repo_root: Path, compute_root: Path) -> List[CheckResult]:
  cache = repo_root / "data_cache"
  return run_checks([
    lambda: check_venv(repo_root),
    lambda: check_torch(repo_root),
    lambda: check_triton_flex(repo_root),
    lambda: check_train_imports(),
    lambda: check_compute_sync(compute_root),
    lambda: check_forward_pass(
      repo_root, attn_backend="sdpa", cross_attn=True, adaln=False, seq_len=128,
    ),
    lambda: check_hydra_compose(repo_root, [
      "model=block_dit",
      "algo=block_diffusion_masked",
      "data=tiny_shakespeare",
      f"data.cache_dir={cache}",
      "model.length=128",
      "block_size=16",
      "model.attn_backend=sdpa",
      "trainer.devices=1",
      "trainer.num_nodes=1",
      "loader.global_batch_size=64",
      "loader.batch_size=64",
    ]),
  ])


def profile_owt(repo_root: Path, compute_root: Path) -> List[CheckResult]:
  cache = repo_root / "data_cache"
  compute_cache = compute_root / "data_cache"
  return run_checks([
    lambda: check_venv(repo_root),
    lambda: check_torch(repo_root),
    lambda: check_triton_flex(repo_root),
    lambda: check_train_imports(),
    lambda: check_compute_sync(compute_root),
    lambda: check_data_cache(cache, scratch=True, pretrain=False),
    lambda: check_data_cache(compute_cache, scratch=True, pretrain=False),
    lambda: check_forward_pass(
      repo_root,
      attn_backend="flex",
      cross_attn=True,
      adaln=False,
      seq_len=128,
      use_cuda=True,
    ),
    lambda: check_dataloader_step(repo_root, "openwebtext-split", cache),
  ])


def profile_owt_pretrain(
    repo_root: Path, compute_root: Path, *, ar: bool,
) -> List[CheckResult]:
  cache = repo_root / "data_cache"
  compute_cache = compute_root / "data_cache"
  ckpt = "ar_noeos_owt.ckpt" if ar else "bd3lm_owt_block1024_pretrain.ckpt"
  dl = (
    "bash slurm_scripts/block_diffusion/download_ar_pretrain.sh"
    if ar else "bash slurm_scripts/block_diffusion/download_mdlm_pretrain.sh"
  )
  return run_checks([
    lambda: check_venv(repo_root),
    lambda: check_torch(repo_root),
    lambda: check_triton_flex(repo_root),
    lambda: check_train_imports(),
    lambda: check_compute_sync(compute_root),
    lambda: check_data_cache(cache, scratch=False, pretrain=True),
    lambda: check_data_cache(compute_cache, scratch=False, pretrain=True),
    lambda: check_checkpoint(cache, ckpt, dl),
    lambda: check_pretrained_load(repo_root, cache / "checkpoints" / ckpt),
    lambda: check_forward_pass(
      repo_root,
      attn_backend="flex",
      cross_attn=False,
      adaln=not ar,
      seq_len=128,
    ),
  ])


def main(argv: Optional[List[str]] = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--profile",
    choices=("tiny", "owt", "owt-pretrain", "owt-ar", "all"),
    default="all",
  )
  parser.add_argument("--repo-root", type=Path, default=None)
  parser.add_argument("--compute-root", type=Path, default=None)
  args = parser.parse_args(argv)

  repo_root = args.repo_root or Path(os.environ.get("REPO_ROOT", ".")).resolve()
  compute_root = args.compute_root or Path(
    os.environ.get("JEDI_COMPUTE_ROOT", "/e/project1/scifi/elsayed3/JEDi")
  )
  src = repo_root / "src"
  if str(src) not in sys.path:
    sys.path.insert(0, str(src))

  profiles = {
    "tiny": profile_tiny,
    "owt": profile_owt,
    "owt-pretrain": lambda r, c: profile_owt_pretrain(r, c, ar=False),
    "owt-ar": lambda r, c: profile_owt_pretrain(r, c, ar=True),
  }

  if args.profile == "all":
    exit_code = 0
    for name in ("tiny", "owt", "owt-pretrain", "owt-ar"):
      print("=" * 60)
      print(f"Profile: {name}")
      print("=" * 60)
      code = print_results(profiles[name](repo_root, compute_root))
      exit_code = max(exit_code, code)
      print()
    return exit_code

  print("=" * 60)
  print(f"Profile: {args.profile}")
  print("=" * 60)
  return print_results(profiles[args.profile](repo_root, compute_root))


if __name__ == "__main__":
  raise SystemExit(main())
