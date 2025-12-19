#!/usr/bin/env python
"""Run a single task from a Slurm job array.

This script is called by the Slurm array job to execute one configuration
from the sweep. It loads the configuration by task ID and runs training.
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


def load_config(configs_file: str, task_id: int) -> Dict[str, Any]:
    """Load configuration for this task ID."""
    with open(configs_file, 'r') as f:
        configs = json.load(f)
    
    if task_id >= len(configs):
        raise ValueError(f"Task ID {task_id} >= number of configs {len(configs)}")
    
    return configs[task_id]


def config_to_overrides(config: Dict[str, Any]) -> List[str]:
    """Convert config dict to Hydra CLI overrides."""
    overrides = []
    for key, value in config.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, str) and any(c in value for c in ' =[]{}'):
            value = f'"{value}"'
        overrides.append(f"{key}={value}")
    return overrides


def run_training(
    config: Dict[str, Any],
    task_id: int,
    sweep_name: str,
    output_dir: str,
) -> int:
    """Run training for this configuration.
    
    Returns:
        Exit code (0 for success)
    """
    run_name = f"{sweep_name}_{task_id:04d}"
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
        f"wandb.group={sweep_name}",
        f"checkpointing.save_dir='{run_dir}'",
    ])
    
    # Get distributed training info from environment
    num_nodes = int(os.environ.get('SLURM_NNODES', 1))
    node_id = int(os.environ.get('SLURM_NODEID', 0))
    
    # Determine GPU configuration
    gpus_env = os.environ.get('SLURM_GPUS_PER_NODE', '')
    if gpus_env:
        # Handle formats like "4" or "gpu:4"
        gpus_per_node = int(gpus_env.split(':')[-1])
    else:
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        gpus_per_node = len(cuda_devices.split(','))
    
    if num_nodes > 1:
        # Multi-node: use torchrun for distributed training
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        
        cmd = [
            "torchrun",
            f"--nnodes={num_nodes}",
            f"--nproc_per_node={gpus_per_node}",
            f"--node_rank={node_id}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            "-m", "discrete_diffusion"
        ] + overrides
        
        overrides.extend([
            f"trainer.num_nodes={num_nodes}",
            f"trainer.devices={gpus_per_node}",
        ])
    else:
        # Single node - for HP tuning, use single GPU per config
        # This maximizes exploration by running configs in parallel
        cmd = [sys.executable, "-m", "discrete_diffusion"] + overrides
        
        # Force single GPU for HP tuning efficiency
        overrides.extend([
            "trainer.devices=1",
            "trainer.num_nodes=1",
        ])
    
    # Log start
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Task ID: {task_id}")
    print(f"Run name: {run_name}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output dir: {run_dir}")
    print(f"{'='*60}\n")
    
    # Run training
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
    
    # Save status
    status = {
        "task_id": task_id,
        "run_name": run_name,
        "exit_code": result.returncode,
        "duration_seconds": duration,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(run_dir / "status.json", 'w') as f:
        json.dump(status, f, indent=2)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Slurm array task")
    parser.add_argument("--configs-file", type=str, required=True,
                       help="Path to JSON file with configurations")
    parser.add_argument("--task-id", type=int, required=True,
                       help="Slurm array task ID")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Sweep output directory")
    parser.add_argument("--sweep-name", type=str, required=True,
                       help="Sweep name for run naming")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.configs_file, args.task_id)
    
    # Run training
    exit_code = run_training(
        config=config,
        task_id=args.task_id,
        sweep_name=args.sweep_name,
        output_dir=args.output_dir,
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

