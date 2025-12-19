"""Sweep runner for distributed hyperparameter tuning.

Supports:
- Single-node execution with sequential or parallel runs
- Multi-node execution via Slurm job arrays
- Checkpoint-based experiment resumption
- Integration with WandB for tracking
- Parallel GPU utilization for efficient HP search
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Union

import yaml

from .sweep_config import SweepConfig, load_sweep_config, estimate_grid_size
from .experiment_tracker import ExperimentTracker


@dataclass
class RunResult:
    """Result from a single experiment run."""
    run_id: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    status: str  # completed, failed, early_stopped
    duration_seconds: float
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None


class SweepRunner:
    """Execute hyperparameter sweeps across configurations."""
    
    def __init__(
        self,
        sweep_config: Union[str, Path, SweepConfig],
        output_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        resume: bool = True,
        dry_run: bool = False,
    ):
        """Initialize sweep runner.
        
        Args:
            sweep_config: Path to sweep YAML or SweepConfig object
            output_dir: Base directory for outputs (default: ./sweep_outputs/<name>)
            wandb_project: WandB project name (default: from config)
            wandb_entity: WandB entity/team (default: from config)
            resume: Whether to skip completed runs on restart
            dry_run: If True, print commands without executing
        """
        if isinstance(sweep_config, (str, Path)):
            self.config = load_sweep_config(sweep_config)
        else:
            self.config = sweep_config
        
        self.output_dir = Path(output_dir or f"./sweep_outputs/{self.config.name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.resume = resume
        self.dry_run = dry_run
        
        # Initialize tracker
        self.tracker = ExperimentTracker(
            self.output_dir / "experiments.json",
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity
        )
    
    def run_sweep(
        self,
        num_runs: Optional[int] = None,
        seed: int = 42,
        checkpoint_path: Optional[str] = None,
        extra_overrides: Optional[List[str]] = None,
        parallel: bool = True,
        num_gpus: Optional[int] = None,
    ) -> List[RunResult]:
        """Run complete hyperparameter sweep.
        
        Args:
            num_runs: Number of runs (for random/lhs methods)
            seed: Random seed for reproducibility
            checkpoint_path: Stage 1 checkpoint for Stage 2 sweeps
            extra_overrides: Additional Hydra overrides for all runs
            parallel: If True, run configs in parallel across GPUs
            num_gpus: Number of GPUs to use (default: auto-detect)
        
        Returns:
            List of RunResult objects
        """
        # Generate configurations
        configs = self.config.generate_configs(num_runs, seed)
        
        # Handle checkpoint requirement
        if self.config.requires_checkpoint and checkpoint_path:
            key = self.config.requires_checkpoint.get('checkpoint_key', 'training.finetune_path')
            for cfg in configs:
                cfg[key] = checkpoint_path
        
        # Detect available GPUs
        if num_gpus is None:
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_devices:
                num_gpus = len(cuda_devices.split(','))
            else:
                # Try to detect via torch
                try:
                    import torch
                    num_gpus = torch.cuda.device_count()
                except ImportError:
                    num_gpus = 1
        
        print(f"\n{'='*60}")
        print(f"Sweep: {self.config.name}")
        print(f"Method: {self.config.method}")
        print(f"Total configurations: {len(configs)}")
        print(f"Parallel execution: {parallel} ({num_gpus} GPUs)")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Save sweep manifest
        self._save_manifest(configs)
        
        # Filter completed runs if resuming
        if self.resume:
            completed = self.tracker.get_completed_runs()
            configs = [c for c in configs if self.config._config_hash(c) not in completed]
            print(f"Remaining configurations after resume: {len(configs)}")
        
        if not configs:
            print("All configurations already completed. Nothing to run.")
            return []
        
        # Run sweep
        if parallel and num_gpus > 1 and not self.dry_run:
            results = self._run_parallel(configs, extra_overrides, num_gpus)
        else:
            results = self._run_sequential(configs, extra_overrides)
        
        # Generate summary
        self._generate_summary(results)
        
        return results
    
    def _run_sequential(
        self,
        configs: List[Dict[str, Any]],
        extra_overrides: Optional[List[str]] = None,
    ) -> List[RunResult]:
        """Run configurations sequentially."""
        results = []
        for idx, config in enumerate(configs):
            run_name = self.config.get_run_name(config, idx)
            print(f"\n[{idx+1}/{len(configs)}] Running: {run_name}")
            
            result = self._run_single(
                config=config,
                run_name=run_name,
                run_idx=idx,
                extra_overrides=extra_overrides,
                gpu_id=0  # Use first GPU
            )
            results.append(result)
            
            # Track result
            self.tracker.log_run(result)
            self._print_run_status(result)
        
        return results
    
    def _run_parallel(
        self,
        configs: List[Dict[str, Any]],
        extra_overrides: Optional[List[str]] = None,
        num_gpus: int = 4,
    ) -> List[RunResult]:
        """Run configurations in parallel across multiple GPUs.
        
        Uses a thread pool to manage concurrent runs, with each run
        assigned to a specific GPU in round-robin fashion.
        """
        print(f"\nStarting parallel execution with {num_gpus} GPUs...")
        
        # Get original CUDA_VISIBLE_DEVICES to map GPU indices
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices:
            gpu_ids = [int(g.strip()) for g in cuda_devices.split(',')]
        else:
            gpu_ids = list(range(num_gpus))
        
        results = []
        results_lock = threading.Lock()
        
        # Create work items: (idx, config, gpu_id)
        work_items = [
            (idx, config, gpu_ids[idx % len(gpu_ids)])
            for idx, config in enumerate(configs)
        ]
        
        def run_on_gpu(work_item):
            idx, config, gpu_id = work_item
            run_name = self.config.get_run_name(config, idx)
            
            result = self._run_single(
                config=config,
                run_name=run_name,
                run_idx=idx,
                extra_overrides=extra_overrides,
                gpu_id=gpu_id
            )
            
            # Thread-safe logging
            with results_lock:
                self.tracker.log_run(result)
                print(f"\n[{len(results)+1}/{len(configs)}] Completed: {run_name} (GPU {gpu_id})")
                self._print_run_status(result)
            
            return result
        
        # Run with thread pool - one thread per GPU
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(run_on_gpu, item): item for item in work_items}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with results_lock:
                        results.append(result)
                except Exception as e:
                    idx, config, gpu_id = futures[future]
                    run_name = self.config.get_run_name(config, idx)
                    print(f"\n[ERROR] {run_name} failed with exception: {e}")
                    results.append(RunResult(
                        run_id=run_name,
                        config=config,
                        metrics={},
                        status="error",
                        duration_seconds=0,
                        error_message=str(e)
                    ))
        
        return results
    
    def _print_run_status(self, result: RunResult):
        """Print status of a completed run."""
        status_emoji = "✓" if result.status == "completed" else "✗"
        print(f"  {status_emoji} Status: {result.status}")
        if result.error_message:
            error_preview = result.error_message.strip().replace('\n', ' ')[:300]
            print(f"  Error: {error_preview}...")
        if result.metrics:
            for k, v in list(result.metrics.items())[:3]:
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
    
    def _run_single(
        self,
        config: Dict[str, Any],
        run_name: str,
        run_idx: int,
        extra_overrides: Optional[List[str]] = None,
        gpu_id: Optional[int] = None,
    ) -> RunResult:
        """Execute a single training run.
        
        Args:
            config: Hyperparameter configuration
            run_name: Name for this run
            run_idx: Index of this run
            extra_overrides: Additional Hydra overrides
            gpu_id: Specific GPU to use (None = use all available)
        """
        run_dir = self.output_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Build Hydra command
        overrides = self.config.config_to_hydra_overrides(config)
        
        # Add run-specific overrides
        overrides.extend([
            f"hydra.run.dir='{run_dir}'",
            f"wandb.name={run_name}",
            f"wandb.group={self.config.name}",
            f"checkpointing.save_dir='{run_dir}'",
            # Force single-GPU for HP tuning efficiency (run many configs in parallel)
            "trainer.devices=1",
            "trainer.num_nodes=1",
        ])
        
        if self.wandb_project:
            overrides.append(f"wandb.project={self.wandb_project}")
        
        if extra_overrides:
            overrides.extend(extra_overrides)
        
        # Build command
        cmd = [sys.executable, "-m", "discrete_diffusion"] + overrides
        
        # Save config
        with open(run_dir / "config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        if self.dry_run:
            gpu_str = f"GPU {gpu_id}" if gpu_id is not None else "all GPUs"
            print(f"  [DRY RUN] Would execute on {gpu_str}: {' '.join(cmd[:5])}...")
            return RunResult(
                run_id=run_name,
                config=config,
                metrics={},
                status="dry_run",
                duration_seconds=0
            )
        
        # Execute training
        start_time = time.time()
        try:
            env = os.environ.copy()
            
            # Set specific GPU if provided (for parallel execution)
            if gpu_id is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            else:
                env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
            
            # Ensure HuggingFace offline mode with correct cache
            # Only set HF_HOME - do NOT set TRANSFORMERS_CACHE (causes cache format issues)
            hf_home = os.path.expanduser('~/.cache/huggingface')
            env['HF_HOME'] = hf_home
            env['HF_HUB_OFFLINE'] = '1'
            env['HYDRA_FULL_ERROR'] = '1'
            # Remove TRANSFORMERS_CACHE if present - it uses old cache format
            env.pop('TRANSFORMERS_CACHE', None)
            
            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir.parent.parent),  # Project root
                env=env,
                capture_output=True,
                text=True,
                timeout=48 * 3600  # 48 hour timeout
            )
            
            duration = time.time() - start_time
            
            # Save logs
            with open(run_dir / "stdout.log", 'w') as f:
                f.write(result.stdout)
            with open(run_dir / "stderr.log", 'w') as f:
                f.write(result.stderr)
            
            if result.returncode != 0:
                return RunResult(
                    run_id=run_name,
                    config=config,
                    metrics=self._extract_metrics(run_dir),
                    status="failed",
                    duration_seconds=duration,
                    error_message=result.stderr[-1000:] if result.stderr else None
                )
            
            return RunResult(
                run_id=run_name,
                config=config,
                metrics=self._extract_metrics(run_dir),
                status="completed",
                duration_seconds=duration,
                checkpoint_path=str(run_dir / "checkpoints" / "last.ckpt")
            )
            
        except subprocess.TimeoutExpired:
            return RunResult(
                run_id=run_name,
                config=config,
                metrics=self._extract_metrics(run_dir),
                status="timeout",
                duration_seconds=48 * 3600,
                error_message="Run exceeded 48 hour timeout"
            )
        except Exception as e:
            return RunResult(
                run_id=run_name,
                config=config,
                metrics={},
                status="error",
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _extract_metrics(self, run_dir: Path) -> Dict[str, float]:
        """Extract final metrics from run directory."""
        metrics = {}
        
        # Try to load from metrics.json if exists
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        return metrics
    
    def _save_manifest(self, configs: List[Dict[str, Any]]):
        """Save sweep manifest with all configurations."""
        manifest = {
            "sweep_name": self.config.name,
            "method": self.config.method,
            "num_configs": len(configs),
            "created_at": datetime.now().isoformat(),
            "metric": self.config.metric,
            "configs": configs
        }
        
        with open(self.output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
    
    def _generate_summary(self, results: List[RunResult]):
        """Generate summary report of sweep results."""
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status in ("failed", "error")]
        
        summary = {
            "sweep_name": self.config.name,
            "total_runs": len(results),
            "completed": len(completed),
            "failed": len(failed),
            "metric": self.config.metric,
        }
        
        if completed and self.config.metric["name"]:
            metric_name = self.config.metric["name"]
            metric_values = [
                r.metrics.get(metric_name) 
                for r in completed 
                if r.metrics.get(metric_name) is not None
            ]
            
            if metric_values:
                summary["best_value"] = (
                    max(metric_values) if self.config.metric["goal"] == "maximize" 
                    else min(metric_values)
                )
                
                # Find best config
                if self.config.metric["goal"] == "maximize":
                    best_run = max(completed, key=lambda r: r.metrics.get(metric_name, float('-inf')))
                else:
                    best_run = min(completed, key=lambda r: r.metrics.get(metric_name, float('inf')))
                
                summary["best_config"] = best_run.config
                summary["best_run_id"] = best_run.run_id
        
        # Save summary
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Sweep Summary: {self.config.name}")
        print(f"{'='*60}")
        print(f"Total runs: {summary['total_runs']}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        if "best_value" in summary:
            print(f"Best {metric_name}: {summary['best_value']:.4f}")
            print(f"Best run: {summary.get('best_run_id', 'N/A')}")
        print(f"{'='*60}\n")
    
    def generate_slurm_array(
        self,
        num_runs: Optional[int] = None,
        seed: int = 42,
        output_script: Optional[str] = None,
        partition: str = "accelerated",
        nodes_per_job: int = 1,
        gpus_per_node: int = 4,
        time_limit: str = "12:00:00",
        memory: str = "256G",
        extra_sbatch: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate Slurm job array script for sweep.
        
        Args:
            num_runs: Number of configurations to generate
            seed: Random seed
            output_script: Output path for script
            partition: Slurm partition
            nodes_per_job: Nodes per array job
            gpus_per_node: GPUs per node
            time_limit: Time limit per job
            memory: Memory per node
            extra_sbatch: Additional SBATCH directives
        
        Returns:
            Path to generated script
        """
        configs = self.config.generate_configs(num_runs, seed)
        
        # Save configurations as JSON for array jobs to load
        configs_file = self.output_dir / "array_configs.json"
        with open(configs_file, 'w') as f:
            json.dump(configs, f, indent=2, default=str)
        
        # Build script
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.config.name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --nodes={nodes_per_job}",
            f"#SBATCH --gres=gpu:{gpus_per_node}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --mem={memory}",
            f"#SBATCH --array=0-{len(configs)-1}%32",  # Max 32 concurrent
            f"#SBATCH --output={self.output_dir}/logs/%A_%a.out",
            f"#SBATCH --error={self.output_dir}/logs/%A_%a.err",
        ]
        
        if extra_sbatch:
            for key, val in extra_sbatch.items():
                script_lines.append(f"#SBATCH --{key}={val}")
        
        script_lines.extend([
            "",
            "set -euo pipefail",
            "",
            f"# Sweep: {self.config.name}",
            f"# Total configurations: {len(configs)}",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "# Load environment",
            "source ~/.bashrc",
            "conda activate diffusion 2>/dev/null || true",
            "export HF_HOME=/home/hk-project-p0023960/hgf_nhz3359/.cache/huggingface",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export HF_DATASETS_OFFLINE=1",
            "",
            "# Project setup",
            f'REPO_ROOT="{self.output_dir.parent.parent}"',
            'cd "${REPO_ROOT}"',
            'export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"',
            "",
            "# Run configuration for this array task",
            f'python -m discrete_diffusion.tuning.run_array_task \\',
            f'    --configs-file "{configs_file}" \\',
            f'    --task-id ${{SLURM_ARRAY_TASK_ID}} \\',
            f'    --output-dir "{self.output_dir}" \\',
            f'    --sweep-name "{self.config.name}"',
        ])
        
        script = "\n".join(script_lines)
        
        # Save script
        output_script = output_script or str(self.output_dir / "run_sweep.sh")
        with open(output_script, 'w') as f:
            f.write(script)
        
        os.chmod(output_script, 0o755)
        
        # Create logs directory
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        print(f"Generated Slurm array script: {output_script}")
        print(f"Configurations saved to: {configs_file}")
        print(f"Submit with: sbatch {output_script}")
        
        return output_script


def main():
    """CLI entry point for sweep runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument("sweep_config", type=str, help="Path to sweep YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num-runs", type=int, default=None, help="Number of runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--generate-slurm", action="store_true", help="Generate Slurm script")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for Stage 2")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run configs in parallel across GPUs (default: True)")
    parser.add_argument("--sequential", action="store_true",
                       help="Run configs sequentially (disables parallel)")
    parser.add_argument("--num-gpus", type=int, default=None,
                       help="Number of GPUs to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    runner = SweepRunner(
        sweep_config=args.sweep_config,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        dry_run=args.dry_run
    )
    
    if args.generate_slurm:
        runner.generate_slurm_array(num_runs=args.num_runs, seed=args.seed)
    else:
        runner.run_sweep(
            num_runs=args.num_runs,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            parallel=not args.sequential,
            num_gpus=args.num_gpus
        )


if __name__ == "__main__":
    main()

