#!/usr/bin/env python
"""Run a single configuration from a sweep configs file.

This script is called by the multi-node parallel orchestrator to execute
one configuration on N GPUs. It loads the configuration by index and
runs training with proper logging.

GPU Configuration:
    --gpus-per-config 1: Single GPU, uses SingleDeviceStrategy
    --gpus-per-config 2+: Multiple GPUs, uses DDP strategy
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def load_config(configs_file: str, config_idx: int) -> Dict[str, Any]:
    """Load configuration by index from configs file."""
    with open(configs_file, 'r') as f:
        configs = json.load(f)
    
    if config_idx >= len(configs):
        raise ValueError(f"Config index {config_idx} >= number of configs {len(configs)}")
    
    return configs[config_idx]


def config_to_overrides(config: Dict[str, Any]) -> List[str]:
    """Convert config dict to Hydra CLI overrides."""
    overrides = []
    for key, value in config.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, str) and any(c in value for c in ' =[]{}\'\"'):
            value = f"'{value}'"
        overrides.append(f"{key}={value}")
    return overrides


def run_training(
    config_data: Dict[str, Any],
    output_dir: str,
    wandb_project: str,
    gpus_per_config: int = 1,
) -> int:
    """Run training for this configuration.
    
    Args:
        config_data: Dict with 'idx', 'run_name', and 'config' keys
        output_dir: Phase output directory
        wandb_project: WandB project name
        gpus_per_config: Number of GPUs to use (1=single-device, >1=DDP)
    
    Returns:
        Exit code (0 for success)
    """
    idx = config_data['idx']
    run_name = config_data['run_name']
    config = config_data['config']
    
    run_dir = Path(output_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    # Build overrides
    overrides = config_to_overrides(config)
    overrides.extend([
        f"hydra.run.dir='{run_dir}'",
        f"wandb.name={run_name}",
        f"wandb.group={run_name.rsplit('_', 4)[0]}",  # Extract sweep name
        f"checkpointing.save_dir='{run_dir}'",
        f"wandb.project={wandb_project}",
        # GPU configuration
        f"trainer.devices={gpus_per_config}",
        "trainer.num_nodes=1",
    ])
    
    # Strategy depends on GPU count
    if gpus_per_config == 1:
        # Single GPU: use SingleDeviceStrategy to avoid DDP overhead
        overrides.append("strategy=single-device")
    # else: use default DDP strategy from config (for multi-GPU)
    
    # Build command
    cmd = [sys.executable, "-m", "discrete_diffusion"] + overrides
    
    # Log start
    start_time = time.time()
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')
    hostname = os.environ.get('SLURMD_NODENAME', os.environ.get('HOSTNAME', 'unknown'))
    strategy = "single-device" if gpus_per_config == 1 else "ddp"
    
    print(f"\n{'='*60}")
    print(f"Config Index: {idx}")
    print(f"Run name: {run_name}")
    print(f"Host: {hostname}")
    print(f"GPUs: {gpus_per_config} (CUDA_VISIBLE_DEVICES={gpu_id})")
    print(f"Strategy: {strategy}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output dir: {run_dir}")
    print(f"{'='*60}\n")
    
    # Save start status
    with open(run_dir / "started.json", 'w') as f:
        json.dump({
            "config_idx": idx,
            "run_name": run_name,
            "hostname": hostname,
            "gpus_per_config": gpus_per_config,
            "cuda_visible_devices": gpu_id,
            "strategy": strategy,
            "started_at": datetime.now().isoformat(),
            "command": cmd[:10]  # First 10 elements
        }, f, indent=2)
    
    # Run training - stream output to console
    result = subprocess.run(
        cmd,
        cwd=str(Path(output_dir).parent.parent),  # Project root
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    
    # Log completion
    duration = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed: {datetime.now().isoformat()}")
    print(f"Duration: {duration/3600:.2f} hours")
    print(f"Exit code: {result.returncode}")
    print(f"{'='*60}\n")
    
    # Save final status
    status = {
        "config_idx": idx,
        "run_name": run_name,
        "hostname": hostname,
        "gpus_per_config": gpus_per_config,
        "cuda_visible_devices": gpu_id,
        "strategy": strategy,
        "exit_code": result.returncode,
        "duration_seconds": duration,
        "started_at": start_time,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(run_dir / "status.json", 'w') as f:
        json.dump(status, f, indent=2)
    
    # Try to extract metrics if available
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        status["has_metrics"] = True
    
    return result.returncode


def main():
    # =========================================================================
    # GPU Binding Note:
    # CUDA_VISIBLE_DEVICES is set by the shell script based on GPUS_PER_CONFIG.
    # For single-GPU: "0" or "1" or "2" or "3"
    # For multi-GPU:  "0,1" or "0,1,2,3" etc.
    # We just log what we received.
    # =========================================================================
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"[GPU Binding] CUDA_VISIBLE_DEVICES={cuda_devices}")
    
    parser = argparse.ArgumentParser(description="Run single config from sweep")
    parser.add_argument("--configs-file", type=str, required=True,
                       help="Path to JSON file with configurations")
    parser.add_argument("--config-idx", type=int, required=True,
                       help="Index of configuration to run")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Phase output directory")
    parser.add_argument("--wandb-project", type=str, default="jepa_hparam_study",
                       help="WandB project name")
    parser.add_argument("--gpus-per-config", type=int, default=1,
                       help="Number of GPUs per config (1=single-device, >1=DDP)")
    
    args = parser.parse_args()
    
    # Log GPU configuration
    print(f"[Config Runner] GPUs per config: {args.gpus_per_config}")
    print(f"[Config Runner] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Load configuration
    config_data = load_config(args.configs_file, args.config_idx)
    
    # Run training
    exit_code = run_training(
        config_data=config_data,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        gpus_per_config=args.gpus_per_config,
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

