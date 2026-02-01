#!/usr/bin/env python
"""Extract metrics from the 350m hyperparameter study.

This script extracts metrics from phase directories that use status.json files
and handles the nested summary structure in the 350m study metrics.

The 350m study metrics have the following structure in status.json and metrics.json:
- Flat metrics: latent/cosine_similarity, trainer/loss, etc.
- Nested summary: metrics["summary"]["metric_name"]["mean"/"std"/"min"/"max"/"last"]

This script extracts:
1. Flat metrics as-is
2. Summary statistics (mean, std, min, max, last) for each metric
3. WandB summary metrics
4. Log-parsed metrics as fallback

Usage:
    python extract_350m_study_metrics.py [STUDY_DIR] [--output-dir OUTPUT_DIR]
    
Example:
    python extract_350m_study_metrics.py /path/to/jepa_350m_full_study_20260102_164904
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd

# Add src to path for imports
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Default study directory
DEFAULT_STUDY_DIR = "/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/sweep_outputs/jepa_350m_full_study_20260102_164904"


def flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten nested metrics dictionary into flat key-value pairs.
    
    Handles the 350m study format where metrics can contain:
    - Flat metrics: {"latent/cosine_similarity": 0.85, ...}
    - Nested summary: {"summary": {"metric_name": {"mean": x, "std": y, ...}}}
    
    Args:
        metrics: Metrics dictionary (possibly nested)
        prefix: Prefix for flattened keys
    
    Returns:
        Flat dictionary with string keys and float values
    """
    flat = {}
    
    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if prefix else key
        
        if key == "summary" and isinstance(value, dict):
            # Handle summary statistics
            for metric_name, stats in value.items():
                if isinstance(stats, dict):
                    for stat_name, stat_value in stats.items():
                        try:
                            flat[f"summary.{metric_name}.{stat_name}"] = float(stat_value)
                        except (ValueError, TypeError):
                            pass
                else:
                    try:
                        flat[f"summary.{metric_name}"] = float(stats)
                    except (ValueError, TypeError):
                        pass
        elif isinstance(value, dict):
            # Recursively flatten other nested dicts
            nested = flatten_metrics(value, f"{full_key}.")
            flat.update(nested)
        elif isinstance(value, (int, float, bool)):
            try:
                flat[full_key] = float(value)
            except (ValueError, TypeError):
                pass
    
    return flat


def extract_metrics_from_status_json(status_file: Path) -> Dict[str, float]:
    """Extract metrics from status.json file.
    
    Args:
        status_file: Path to status.json
    
    Returns:
        Flattened metrics dictionary
    """
    try:
        with open(status_file) as f:
            status = json.load(f)
        
        metrics = status.get("metrics", {})
        return flatten_metrics(metrics)
    except (json.JSONDecodeError, IOError):
        return {}


def extract_metrics_from_metrics_json(run_dir: Path) -> Dict[str, float]:
    """Extract metrics from metrics.json file.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Flattened metrics dictionary
    """
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return {}
    
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
        return flatten_metrics(metrics)
    except (json.JSONDecodeError, IOError):
        return {}


def extract_metrics_from_wandb(run_dir: Path) -> Dict[str, float]:
    """Extract metrics from WandB summary files.
    
    Finds the most complete wandb-summary.json file in the run directory.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    wandb_dir = run_dir / "wandb"
    
    if not wandb_dir.exists():
        return metrics
    
    # Find the best (most complete) wandb summary file
    best_summary = None
    best_metric_count = 0
    
    for run_subdir in wandb_dir.iterdir():
        if run_subdir.is_dir() and run_subdir.name.startswith("run-"):
            summary_file = run_subdir / "files" / "wandb-summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        wandb_summary = json.load(f)
                    # Count non-internal metrics
                    metric_count = len([
                        k for k in wandb_summary.keys() 
                        if not k.startswith("_") and not k.startswith("trainer/")
                    ])
                    if metric_count > best_metric_count:
                        best_summary = wandb_summary
                        best_metric_count = metric_count
                except (json.JSONDecodeError, IOError):
                    continue
    
    if best_summary:
        for key, value in best_summary.items():
            # Skip internal WandB keys
            if key.startswith("_"):
                continue
            try:
                if isinstance(value, (int, float)):
                    metrics[f"wandb.{key}"] = float(value)
                elif isinstance(value, bool):
                    metrics[f"wandb.{key}"] = float(value)
            except (ValueError, TypeError):
                pass
    
    return metrics


def extract_metrics_from_logs(run_dir: Path) -> Dict[str, float]:
    """Extract metrics from training log files.
    
    Parses stdout.log and stderr.log for metric values.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    for log_file in ["stdout.log", "stderr.log"]:
        log_path = run_dir / log_file
        if not log_path.exists():
            continue
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Pattern 1: PyTorch Lightning validation log format
            # Epoch 0, global step 5000: 'val/nll' reached 0.72412
            pattern1 = r"'(val/[a-z_/]+|train/[a-z_/]+|latent/[a-z_/]+)'\s+reached\s+([\d.eE+-]+)"
            for match in re.finditer(pattern1, content):
                metric_name = match.group(1)
                try:
                    metric_val = float(match.group(2))
                    metrics[f"log.{metric_name}"] = metric_val
                except ValueError:
                    continue
            
            # Pattern 2: Simple key=value format
            pattern2 = r"(val[/_][a-z_/]+|train[/_][a-z_/]+|latent[/_][a-z_/]+)[=:]\s*([\d.eE+-]+)"
            for match in re.finditer(pattern2, content):
                metric_name = match.group(1).replace('_', '/')
                try:
                    metric_val = float(match.group(2))
                    if f"log.{metric_name}" not in metrics:
                        metrics[f"log.{metric_name}"] = metric_val
                except ValueError:
                    continue
            
            # Pattern 3: JSON-like format in logs
            pattern3 = r'"(val/[^"]+|train/[^"]+|latent/[^"]+)":\s*([\d.eE+-]+)'
            for match in re.finditer(pattern3, content):
                metric_name = match.group(1)
                try:
                    metric_val = float(match.group(2))
                    if f"log.{metric_name}" not in metrics:
                        metrics[f"log.{metric_name}"] = metric_val
                except ValueError:
                    continue
                    
        except (IOError, UnicodeDecodeError):
            continue
    
    return metrics


def extract_all_metrics_from_run(run_dir: Path) -> Dict[str, float]:
    """Extract all available metrics from a run directory.
    
    Combines metrics from:
    1. status.json (primary source)
    2. metrics.json (comprehensive metrics)
    3. WandB summary files
    4. Log files (fallback)
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Combined metrics dictionary
    """
    all_metrics = {}
    
    # 1. Extract from status.json
    status_file = run_dir / "status.json"
    if status_file.exists():
        status_metrics = extract_metrics_from_status_json(status_file)
        all_metrics.update(status_metrics)
    
    # 2. Extract from metrics.json (may have more complete data)
    metrics_json = extract_metrics_from_metrics_json(run_dir)
    for key, value in metrics_json.items():
        if key not in all_metrics:
            all_metrics[key] = value
    
    # 3. Extract from WandB
    wandb_metrics = extract_metrics_from_wandb(run_dir)
    for key, value in wandb_metrics.items():
        if key not in all_metrics:
            all_metrics[key] = value
    
    # 4. Extract from logs (lowest priority)
    log_metrics = extract_metrics_from_logs(run_dir)
    for key, value in log_metrics.items():
        if key not in all_metrics:
            all_metrics[key] = value
    
    return all_metrics


def load_phase_results(phase_dir: Union[str, Path]) -> pd.DataFrame:
    """Load results from a phase directory.
    
    Args:
        phase_dir: Path to phase directory (e.g., phase1a_encoder)
    
    Returns:
        DataFrame with one row per run, including config and metrics
    """
    phase_dir = Path(phase_dir)
    configs_file = phase_dir / "configs.json"
    
    if not configs_file.exists():
        raise FileNotFoundError(f"No configs.json found in {phase_dir}")
    
    # Load configurations
    try:
        with open(configs_file) as f:
            configs = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {configs_file}: {e}")
    
    rows = []
    for cfg_data in configs:
        run_name = cfg_data['run_name']
        run_dir = phase_dir / "runs" / run_name
        status_file = run_dir / "status.json"
        
        # Build base row
        row = {
            "run_id": run_name,
            "phase": phase_dir.name,
            "config_idx": cfg_data.get("idx"),
        }
        
        # Add config parameters
        config = cfg_data.get("config", {})
        for key, value in config.items():
            row[f"config.{key}"] = value
        
        if status_file.exists():
            try:
                with open(status_file) as f:
                    status = json.load(f)
                
                # Add status information
                row["status"] = "completed" if status.get("exit_code", 1) == 0 else "failed"
                row["exit_code"] = status.get("exit_code", 1)
                row["duration_seconds"] = status.get("duration_seconds")
                row["duration_hours"] = row["duration_seconds"] / 3600 if row["duration_seconds"] else None
                row["hostname"] = status.get("hostname")
                row["gpus_per_config"] = status.get("gpus_per_config")
                row["checkpoint_path"] = status.get("checkpoint_path")
                row["started_at"] = status.get("started_at")
                row["completed_at"] = status.get("completed_at")
                
                # Extract all metrics
                all_metrics = extract_all_metrics_from_run(run_dir)
                for key, value in all_metrics.items():
                    row[f"metric.{key}"] = value
                    
            except (json.JSONDecodeError, IOError) as e:
                row["status"] = "error"
                row["error_message"] = str(e)
        else:
            row["status"] = "missing"
            row["exit_code"] = None
            row["duration_seconds"] = None
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def load_all_phases(study_dir: Union[str, Path]) -> pd.DataFrame:
    """Load results from all phases in a study directory.
    
    Args:
        study_dir: Path to study directory containing phase subdirectories
    
    Returns:
        Combined DataFrame with all phases
    """
    study_dir = Path(study_dir)
    
    # Phase directories to process (in order)
    phase_names = [
        "phase1a_encoder",
        "phase1b_arch_validation",
        "phase2a_predictor",
        "phase2b_training",
        "phase2c_ema",
        "phase3_decoder",
    ]
    
    all_dfs = []
    for phase_name in phase_names:
        phase_dir = study_dir / phase_name
        if phase_dir.exists() and (phase_dir / "configs.json").exists():
            try:
                df = load_phase_results(phase_dir)
                all_dfs.append(df)
                completed = len(df[df["status"] == "completed"])
                print(f"Loaded {len(df)} runs from {phase_name} ({completed} completed)")
            except Exception as e:
                print(f"Warning: Could not load {phase_name}: {e}")
                continue
    
    if not all_dfs:
        raise ValueError(f"No phase data found in {study_dir}")
    
    # Combine all phases
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def compute_phase_summary(df: pd.DataFrame, phase_name: str) -> Dict:
    """Compute summary statistics for a phase.
    
    Args:
        df: DataFrame with phase data
        phase_name: Name of the phase
    
    Returns:
        Summary dictionary
    """
    completed = df[df["status"] == "completed"]
    
    summary = {
        "phase": phase_name,
        "total_runs": len(df),
        "completed": len(completed),
        "failed": len(df[df["status"] == "failed"]),
        "missing": len(df[df["status"] == "missing"]),
        "completion_rate": len(completed) / len(df) * 100 if len(df) > 0 else 0,
    }
    
    # Compute duration statistics
    if "duration_hours" in df.columns:
        durations = completed["duration_hours"].dropna()
        if len(durations) > 0:
            summary["duration_hours"] = {
                "mean": float(durations.mean()),
                "std": float(durations.std()) if len(durations) > 1 else 0,
                "min": float(durations.min()),
                "max": float(durations.max()),
                "total": float(durations.sum()),
            }
    
    # Compute metric statistics for key metrics
    key_metrics = [
        "metric.latent/cosine_similarity",
        "metric.latent/pred_loss",
        "metric.latent/reg_loss",
        "metric.trainer/loss",
        "metric.val/nll",
        "metric.val/ppl",
        "metric.latent_eval/cosine_sim",
        "metric.summary.latent/cosine_similarity.mean",
        "metric.summary.latent/cosine_similarity.max",
    ]
    
    metric_stats = {}
    for col in key_metrics:
        if col in completed.columns:
            values = completed[col].dropna()
            if len(values) > 0:
                metric_stats[col.replace("metric.", "")] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()) if len(values) > 1 else 0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "count": len(values),
                }
    
    if metric_stats:
        summary["key_metrics"] = metric_stats
    
    # Find best configurations
    if len(completed) > 0:
        # Best by cosine similarity (maximize)
        if "metric.latent/cosine_similarity" in completed.columns:
            best_idx = completed["metric.latent/cosine_similarity"].idxmax()
            if pd.notna(best_idx):
                best_row = completed.loc[best_idx]
                summary["best_by_cosine_similarity"] = {
                    "run_id": best_row["run_id"],
                    "value": float(best_row["metric.latent/cosine_similarity"]),
                    "config_idx": best_row.get("config_idx"),
                }
        
        # Best by val/nll (minimize)
        if "metric.val/nll" in completed.columns:
            nll_values = completed["metric.val/nll"].dropna()
            if len(nll_values) > 0:
                best_idx = nll_values.idxmin()
                best_row = completed.loc[best_idx]
                summary["best_by_val_nll"] = {
                    "run_id": best_row["run_id"],
                    "value": float(best_row["metric.val/nll"]),
                    "config_idx": best_row.get("config_idx"),
                }
    
    return summary


def extract_study_metrics(
    study_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Extract all metrics from a study and save to files.
    
    Args:
        study_dir: Path to study directory
        output_dir: Directory to save extracted metrics (default: study_dir/extracted_metrics)
    
    Returns:
        Combined DataFrame with all runs
    """
    study_dir = Path(study_dir)
    
    if output_dir is None:
        output_dir = study_dir / "extracted_metrics"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting metrics from: {study_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Load all phases
    combined_df = load_all_phases(study_dir)
    
    # Save combined results
    combined_csv = output_dir / "all_phases_combined.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nSaved combined results: {combined_csv}")
    print(f"Total runs: {len(combined_df)}")
    
    # Try to save parquet
    try:
        combined_parquet = output_dir / "all_phases_combined.parquet"
        combined_df.to_parquet(combined_parquet, index=False)
        print(f"Parquet: {combined_parquet}")
    except Exception as e:
        print(f"Warning: Could not save Parquet: {e}")
    
    # Process each phase
    phase_summaries = {}
    for phase_name in combined_df["phase"].unique():
        phase_df = combined_df[combined_df["phase"] == phase_name]
        
        # Save phase CSV
        phase_csv = output_dir / f"{phase_name}.csv"
        phase_df.to_csv(phase_csv, index=False)
        
        # Save completed runs only
        completed = phase_df[phase_df["status"] == "completed"]
        if len(completed) > 0:
            completed_csv = output_dir / f"{phase_name}_completed.csv"
            completed.to_csv(completed_csv, index=False)
        
        # Compute and save summary
        summary = compute_phase_summary(phase_df, phase_name)
        phase_summaries[phase_name] = summary
        
        summary_file = output_dir / f"{phase_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{phase_name}:")
        print(f"  Total: {summary['total_runs']}, Completed: {summary['completed']}, "
              f"Failed: {summary['failed']}")
        if "key_metrics" in summary and "latent/cosine_similarity" in summary["key_metrics"]:
            cs = summary["key_metrics"]["latent/cosine_similarity"]
            print(f"  Cosine similarity: {cs['mean']:.4f} +/- {cs['std']:.4f} "
                  f"(range: {cs['min']:.4f} - {cs['max']:.4f})")
    
    # Generate overall study summary
    study_summary = {
        "study_dir": str(study_dir),
        "total_runs": len(combined_df),
        "completed": len(combined_df[combined_df["status"] == "completed"]),
        "failed": len(combined_df[combined_df["status"] == "failed"]),
        "missing": len(combined_df[combined_df["status"] == "missing"]),
        "phases": phase_summaries,
    }
    
    summary_file = output_dir / "study_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(study_summary, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("Extraction complete!")
    print(f"Study summary: {summary_file}")
    
    return combined_df


def print_metrics_summary(df: pd.DataFrame):
    """Print summary of available metrics."""
    metric_cols = sorted([c for c in df.columns if c.startswith("metric.")])
    
    print(f"\n{'=' * 70}")
    print("METRICS SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total runs: {len(df)}")
    print(f"Completed: {len(df[df['status'] == 'completed'])}")
    print(f"Failed: {len(df[df['status'] == 'failed'])}")
    print(f"Missing: {len(df[df['status'] == 'missing'])}")
    
    # Group metrics by type
    flat_metrics = [c for c in metric_cols if not c.startswith("metric.summary.") 
                    and not c.startswith("metric.wandb.") and not c.startswith("metric.log.")]
    summary_metrics = [c for c in metric_cols if c.startswith("metric.summary.")]
    wandb_metrics = [c for c in metric_cols if c.startswith("metric.wandb.")]
    log_metrics = [c for c in metric_cols if c.startswith("metric.log.")]
    
    print(f"\nAvailable metrics: {len(metric_cols)} total")
    print(f"  - Flat metrics: {len(flat_metrics)}")
    print(f"  - Summary stats: {len(summary_metrics)}")
    print(f"  - WandB metrics: {len(wandb_metrics)}")
    print(f"  - Log metrics: {len(log_metrics)}")
    
    if flat_metrics:
        print(f"\nFlat metrics ({len(flat_metrics)}):")
        for col in flat_metrics[:15]:
            print(f"  {col.replace('metric.', '')}")
        if len(flat_metrics) > 15:
            print(f"  ... and {len(flat_metrics) - 15} more")
    
    if summary_metrics:
        # Group by base metric name
        base_names = set()
        for col in summary_metrics:
            parts = col.replace("metric.summary.", "").rsplit(".", 1)
            if len(parts) == 2:
                base_names.add(parts[0])
        print(f"\nSummary statistics available for {len(base_names)} metrics:")
        for name in sorted(base_names)[:10]:
            print(f"  {name}")
        if len(base_names) > 10:
            print(f"  ... and {len(base_names) - 10} more")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract metrics from 350m hyperparameter study"
    )
    parser.add_argument(
        "study_dir",
        nargs="?",
        default=DEFAULT_STUDY_DIR,
        help=f"Path to study directory (default: {DEFAULT_STUDY_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: study_dir/extracted_metrics)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed metrics summary"
    )
    
    args = parser.parse_args()
    
    # Extract metrics
    df = extract_study_metrics(args.study_dir, args.output_dir)
    
    # Print summary
    print_metrics_summary(df)
    
    if args.verbose:
        print(f"\n{'=' * 70}")
        print("CONFIG COLUMNS")
        print(f"{'=' * 70}")
        config_cols = sorted([c for c in df.columns if c.startswith("config.")])
        for col in config_cols:
            print(f"  {col}")
