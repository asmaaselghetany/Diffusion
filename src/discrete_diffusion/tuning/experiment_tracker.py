"""Experiment tracking and result aggregation for hyperparameter sweeps.

Provides persistent tracking of experiment runs with metrics,
configurations, and status for analysis and resumption.
"""

from __future__ import annotations
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd


class ExperimentTracker:
    """Track experiments and aggregate results across sweep runs."""
    
    def __init__(
        self,
        db_path: Union[str, Path],
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        """Initialize experiment tracker.
        
        Args:
            db_path: Path to JSON database file
            wandb_project: Optional WandB project for syncing
            wandb_entity: Optional WandB entity
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # Load existing database
        self._load_db()
    
    def _load_db(self):
        """Load database from disk."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                self._db = json.load(f)
        else:
            self._db = {
                "runs": [],
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
    
    def _save_db(self):
        """Save database to disk."""
        self._db["updated_at"] = datetime.now().isoformat()
        with open(self.db_path, 'w') as f:
            json.dump(self._db, f, indent=2, default=str)
    
    def log_run(self, result: 'RunResult'):
        """Log a completed run result.
        
        Args:
            result: RunResult object from sweep runner
        """
        from .sweep_runner import RunResult
        
        run_entry = {
            "run_id": result.run_id,
            "config": result.config,
            "metrics": result.metrics,
            "status": result.status,
            "duration_seconds": result.duration_seconds,
            "checkpoint_path": result.checkpoint_path,
            "error_message": result.error_message,
            "logged_at": datetime.now().isoformat()
        }
        
        # Update or append
        existing_idx = next(
            (i for i, r in enumerate(self._db["runs"]) if r["run_id"] == result.run_id),
            None
        )
        
        if existing_idx is not None:
            self._db["runs"][existing_idx] = run_entry
        else:
            self._db["runs"].append(run_entry)
        
        self._save_db()
        
        # Optional WandB logging
        if self.wandb_project:
            self._log_to_wandb(run_entry)
    
    def _log_to_wandb(self, run_entry: Dict[str, Any]):
        """Log run entry to WandB."""
        try:
            import wandb
            
            # Log as a summary run
            run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=run_entry["run_id"],
                config=run_entry["config"],
                reinit=True
            )
            
            wandb.log(run_entry["metrics"])
            wandb.log({
                "status": run_entry["status"],
                "duration_seconds": run_entry["duration_seconds"]
            })
            
            run.finish()
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Failed to log to WandB: {e}")
    
    def get_completed_runs(self) -> Set[str]:
        """Get set of completed run config hashes."""
        import hashlib
        
        completed = set()
        for run in self._db["runs"]:
            if run["status"] == "completed":
                config_str = json.dumps(sorted(run["config"].items()), default=str)
                config_hash = hashlib.md5(config_str.encode()).hexdigest()
                completed.add(config_hash)
        
        return completed
    
    def get_best_run(self, metric_name: str, goal: str = "maximize") -> Optional[Dict[str, Any]]:
        """Get the best run by metric.
        
        Args:
            metric_name: Name of metric to optimize
            goal: 'maximize' or 'minimize'
        
        Returns:
            Best run entry or None if no completed runs
        """
        completed = [r for r in self._db["runs"] if r["status"] == "completed"]
        if not completed:
            return None
        
        def get_metric(run):
            return run.get("metrics", {}).get(metric_name, float('-inf') if goal == "maximize" else float('inf'))
        
        if goal == "maximize":
            return max(completed, key=get_metric)
        return min(completed, key=get_metric)
    
    def get_top_k_runs(self, metric_name: str, k: int = 5, goal: str = "maximize") -> List[Dict[str, Any]]:
        """Get top-k runs by metric.
        
        Args:
            metric_name: Name of metric to optimize
            k: Number of top runs to return
            goal: 'maximize' or 'minimize'
        
        Returns:
            List of top-k run entries
        """
        completed = [r for r in self._db["runs"] if r["status"] == "completed"]
        if not completed:
            return []
        
        def get_metric(run):
            return run.get("metrics", {}).get(metric_name, float('-inf') if goal == "maximize" else float('inf'))
        
        sorted_runs = sorted(completed, key=get_metric, reverse=(goal == "maximize"))
        return sorted_runs[:k]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert runs to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with one row per run
        """
        rows = []
        for run in self._db["runs"]:
            row = {
                "run_id": run["run_id"],
                "status": run["status"],
                "duration_seconds": run["duration_seconds"],
                "logged_at": run["logged_at"]
            }
            
            # Flatten config
            for key, value in run.get("config", {}).items():
                row[f"config.{key}"] = value
            
            # Flatten metrics
            for key, value in run.get("metrics", {}).items():
                row[f"metric.{key}"] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all runs.
        
        Returns:
            Dictionary with summary statistics
        """
        total = len(self._db["runs"])
        completed = sum(1 for r in self._db["runs"] if r["status"] == "completed")
        failed = sum(1 for r in self._db["runs"] if r["status"] in ("failed", "error"))
        early_stopped = sum(1 for r in self._db["runs"] if r["status"] == "early_stopped")
        
        durations = [r["duration_seconds"] for r in self._db["runs"] if r["duration_seconds"]]
        
        return {
            "total_runs": total,
            "completed": completed,
            "failed": failed,
            "early_stopped": early_stopped,
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "total_compute_seconds": sum(durations)
        }
    
    def export_results(self, output_path: Union[str, Path], format: str = "csv"):
        """Export results to file.
        
        Args:
            output_path: Output file path
            format: 'csv', 'json', or 'parquet'
        """
        df = self.to_dataframe()
        output_path = Path(output_path)
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"Exported {len(df)} runs to {output_path}")


def aggregate_sweep_results(
    sweep_dirs: List[Union[str, Path]],
    output_dir: Union[str, Path]
) -> pd.DataFrame:
    """Aggregate results from multiple sweep directories.
    
    Args:
        sweep_dirs: List of sweep output directories
        output_dir: Directory to save aggregated results
    
    Returns:
        Combined DataFrame with all results
    """
    all_dfs = []
    
    for sweep_dir in sweep_dirs:
        sweep_dir = Path(sweep_dir)
        db_path = sweep_dir / "experiments.json"
        
        if db_path.exists():
            tracker = ExperimentTracker(db_path)
            df = tracker.to_dataframe()
            df["sweep_dir"] = str(sweep_dir)
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Save aggregated results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    combined.to_csv(output_dir / "aggregated_results.csv", index=False)
    combined.to_parquet(output_dir / "aggregated_results.parquet", index=False)
    
    print(f"Aggregated {len(combined)} runs from {len(all_dfs)} sweeps")
    
    return combined

