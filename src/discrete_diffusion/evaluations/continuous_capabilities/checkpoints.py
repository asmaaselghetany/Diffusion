"""Checkpoint discovery and loading helpers for continuous capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra.utils
import torch
from omegaconf import DictConfig, OmegaConf


CONTINUOUS_ALGO_TARGET = "discrete_diffusion.algorithms.continuous_embedding_diffusion.ContinuousEmbeddingDiffusion"


@dataclass
class CheckpointRecord:
    """Metadata for an evaluation checkpoint candidate."""

    checkpoint_path: Path
    run_dir: Path
    run_name: str
    metadata: Dict[str, Any]
    is_continuous: bool
    is_decoder_only: bool
    skip_reason: Optional[str] = None



def _to_plain(obj: Any) -> Any:
    if OmegaConf.is_config(obj):
        # Keep interpolation nodes unresolved here; capability metadata does not
        # require expanded loader/trainer arithmetic resolvers.
        return OmegaConf.to_container(obj, resolve=False)
    return obj



def parse_checkpoint_metadata(config: DictConfig | Dict[str, Any]) -> Dict[str, Any]:
    """Extract lightweight, comparable metadata from a Hydra config."""
    cfg = _to_plain(config)
    algo_cfg = cfg.get("algo", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    return {
        "algo_target": algo_cfg.get("_target_"),
        "algo_name": algo_cfg.get("name"),
        "algo_stage": int(algo_cfg.get("stage", -1)) if algo_cfg.get("stage") is not None else None,
        "algo_parameterization": algo_cfg.get("parameterization"),
        "model_parameterization": model_cfg.get("parameterization"),
        "model_length": int(model_cfg.get("length", -1)) if model_cfg.get("length") is not None else None,
        "sampling_method": cfg.get("sampling", {}).get("sampling_method") if isinstance(cfg, dict) else None,
    }



def _load_hydra_run_config(run_dir: Path) -> Optional[DictConfig]:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    return OmegaConf.load(cfg_path)



def discover_checkpoint_paths(checkpoint_dir: str | Path, checkpoint_pattern: str) -> List[Path]:
    """Discover checkpoints under a root directory using a glob pattern."""
    root = Path(checkpoint_dir)
    search_path = str(root / checkpoint_pattern)
    paths = sorted(Path(p) for p in glob(search_path, recursive=True))
    return [p for p in paths if p.is_file()]



def build_checkpoint_record(checkpoint_path: Path) -> CheckpointRecord:
    """Build a record for one checkpoint without loading the model weights."""
    run_dir = checkpoint_path.parent.parent
    run_name = run_dir.name

    run_cfg = _load_hydra_run_config(run_dir)
    if run_cfg is None:
        metadata = {
            "algo_target": None,
            "algo_name": None,
            "algo_stage": None,
            "algo_parameterization": None,
            "model_parameterization": None,
            "model_length": None,
            "sampling_method": None,
        }
        return CheckpointRecord(
            checkpoint_path=checkpoint_path,
            run_dir=run_dir,
            run_name=run_name,
            metadata=metadata,
            is_continuous=False,
            is_decoder_only=False,
            skip_reason="missing .hydra/config.yaml",
        )

    metadata = parse_checkpoint_metadata(run_cfg)
    is_continuous = metadata.get("algo_target") == CONTINUOUS_ALGO_TARGET
    stage = metadata.get("algo_stage")
    is_decoder_only = stage == 2

    skip_reason = None
    if not is_continuous:
        skip_reason = (
            "non-continuous algorithm target "
            f"({metadata.get('algo_target')})"
        )

    return CheckpointRecord(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        run_name=run_name,
        metadata=metadata,
        is_continuous=is_continuous,
        is_decoder_only=is_decoder_only,
        skip_reason=skip_reason,
    )



def discover_checkpoint_records(
    checkpoint_dir: str | Path,
    checkpoint_pattern: str,
    *,
    max_checkpoints: Optional[int] = None,
    continuous_only: bool = True,
) -> List[CheckpointRecord]:
    """Discover and parse checkpoint records for evaluation."""
    paths = discover_checkpoint_paths(checkpoint_dir, checkpoint_pattern)
    records = [build_checkpoint_record(path) for path in paths]

    if continuous_only:
        records = [r for r in records if r.is_continuous]

    if max_checkpoints is not None and max_checkpoints > 0:
        records = records[:max_checkpoints]

    return records



def load_trainer_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    """Load Lightning trainer, tokenizer, and config from a checkpoint."""
    from ...data import get_tokenizer

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if "hyper_parameters" not in ckpt:
        raise ValueError("Checkpoint does not contain 'hyper_parameters'.")
    if "config" not in ckpt["hyper_parameters"]:
        raise ValueError("Checkpoint hyper_parameters missing 'config'.")

    model_config = ckpt["hyper_parameters"]["config"]
    if not OmegaConf.is_config(model_config):
        model_config = OmegaConf.create(model_config)

    tokenizer = get_tokenizer(model_config)

    algo_target = model_config.algo._target_
    algo_cls = hydra.utils.get_class(algo_target)

    # Avoid Lightning's internal torch.load path here because some cluster
    # environments default to `weights_only=True` in ways that break older
    # checkpoint payloads containing OmegaConf objects.
    model = algo_cls(config=model_config, tokenizer=tokenizer)
    state_dict = ckpt.get("state_dict", None)
    if state_dict is None:
        raise ValueError("Checkpoint missing 'state_dict'.")
    model.load_state_dict(state_dict, strict=False)

    ema_loaded = False
    if getattr(model, "ema", None) is not None and "ema" in ckpt:
        model.ema.load_state_dict(ckpt["ema"])
        ema_loaded = True

    model.to(device)
    if ema_loaded and hasattr(model, "_eval_mode"):
        # Stage-1/2 continuous checkpoints store EMA separately from state_dict.
        # Use EMA weights at inference to match training-time validation behavior.
        if hasattr(model.ema, "move_shadow_params_to_device"):
            model.ema.move_shadow_params_to_device(device)
        model._eval_mode()
    else:
        model.eval()

    return model, tokenizer, model_config


__all__ = [
    "CheckpointRecord",
    "CONTINUOUS_ALGO_TARGET",
    "parse_checkpoint_metadata",
    "discover_checkpoint_paths",
    "build_checkpoint_record",
    "discover_checkpoint_records",
    "load_trainer_from_checkpoint",
]
