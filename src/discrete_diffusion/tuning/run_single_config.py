#!/usr/bin/env python
"""Run a single configuration from a sweep configs file.

This script is called by the multi-node parallel orchestrator to execute
one configuration on N GPUs. It loads the configuration by index and
runs training with proper logging and metrics extraction.

GPU Configuration:
    --gpus-per-config 1: Single GPU, uses SingleDeviceStrategy
    --gpus-per-config 2+: Multiple GPUs, uses DDP strategy
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def extract_metrics_from_logs(run_dir: Path) -> Dict[str, float]:
    """Extract metrics from training logs.
    
    Parses stdout/stderr logs for common metric patterns like:
    - 'val/nll': 0.7241
    - Epoch 0, global step 5000: 'val/nll' reached 0.72412
    - metric_name=value format
    """
    metrics = {}
    
    # Try to load from metrics.json first (if training wrote it)
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            return metrics
        except (json.JSONDecodeError, IOError):
            pass
    
    # Parse from logs
    for log_file in ["stdout.log", "stderr.log"]:
        log_path = run_dir / log_file
        if not log_path.exists():
            # Also check parent logs directory
            parent_log = run_dir.parent.parent / "logs" / f"config_{run_dir.name.split('_')[0]}.out"
            if parent_log.exists():
                log_path = parent_log
            else:
                continue
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Pattern 1: PyTorch Lightning validation log format
            # Epoch 0, global step 5000: 'val/nll' reached 0.72412
            pattern1 = r"'(val/[a-z_]+|train/[a-z_]+|latent/[a-z_]+)'\s+reached\s+([\d.]+)"
            for match in re.finditer(pattern1, content):
                metric_name = match.group(1)
                metric_val = float(match.group(2))
                # Keep the latest (best) value
                if metric_name not in metrics or (
                    'nll' in metric_name and metric_val < metrics[metric_name]
                ) or (
                    'accuracy' in metric_name and metric_val > metrics[metric_name]
                ):
                    metrics[metric_name] = metric_val
            
            # Pattern 2: Simple key=value format
            # val/nll=0.7241 or val_nll: 0.7241
            pattern2 = r"(val[/_][a-z_]+|train[/_][a-z_]+|latent[/_][a-z_]+)[=:]\s*([\d.]+)"
            for match in re.finditer(pattern2, content):
                metric_name = match.group(1).replace('_', '/')
                metric_val = float(match.group(2))
                if metric_name not in metrics:
                    metrics[metric_name] = metric_val
            
            # Pattern 3: WandB summary log format
            pattern3 = r'"(val/[^"]+|train/[^"]+)":\s*([\d.]+)'
            for match in re.finditer(pattern3, content):
                metric_name = match.group(1)
                metric_val = float(match.group(2))
                if metric_name not in metrics:
                    metrics[metric_name] = metric_val
                    
        except (IOError, UnicodeDecodeError):
            continue
    
    return metrics


def find_checkpoint(run_dir: Path) -> Optional[str]:
    """Find the best available checkpoint in run directory."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    
    # Priority order: best.ckpt > last.ckpt > any .ckpt
    for name in ["best.ckpt", "last.ckpt"]:
        ckpt = checkpoints_dir / name
        if ckpt.exists():
            return str(ckpt)
    
    # Fallback to any checkpoint
    ckpts = list(checkpoints_dir.glob("*.ckpt"))
    if ckpts:
        # Sort by modification time, newest first
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(ckpts[0])
    
    return None


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
    
    # Check if already completed successfully
    status_file = run_dir / "status.json"
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            exit_code = status.get('exit_code', 1)
            if exit_code == 0:
                print(f"\n{'='*60}")
                print(f"Config {idx} ({run_name}) already completed successfully.")
                print(f"Skipping re-execution.")
                print(f"{'='*60}\n")
                return 0
        except (json.JSONDecodeError, KeyError):
            # If status file is corrupted, continue with execution
            pass
    
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
    
    # Run training - capture output for metrics extraction
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    
    with open(stdout_log, 'w') as stdout_f, open(stderr_log, 'w') as stderr_f:
        result = subprocess.run(
            cmd,
            cwd=str(Path(output_dir).parent.parent),  # Project root
            stdout=stdout_f,
            stderr=stderr_f,
        )
    
    # Also print to console for real-time monitoring
    print(f"\n--- Training output summary (last 50 lines of stderr) ---")
    try:
        with open(stderr_log, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line, end='')
    except IOError:
        pass
    print(f"\n--- End training output ---\n")
    
    # Log completion
    duration = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed: {datetime.now().isoformat()}")
    print(f"Duration: {duration/3600:.2f} hours")
    print(f"Exit code: {result.returncode}")
    print(f"{'='*60}\n")
    
    # Extract metrics from logs
    metrics = extract_metrics_from_logs(run_dir)
    
    # Find checkpoint
    checkpoint_path = find_checkpoint(run_dir)
    
    # Save final status with metrics
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
        "completed_at": datetime.now().isoformat(),
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
    }
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    # Also save metrics separately for easy access
    if metrics:
        with open(run_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Extracted metrics: {metrics}")
    
    return result.returncode


def main():
    # =========================================================================
    # GPU Binding Note:
    # CUDA_VISIBLE_DEVICES is set by the shell script based on GPUS_PER_CONFIG.
    # For single-GPU: "0" or "1" or "2" or "3"
    # For multi-GPU:  "0,1" or "0,1,2,3" etc.
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
