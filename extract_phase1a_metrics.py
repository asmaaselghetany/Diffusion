#!/usr/bin/env python
"""Extract metrics from all phases of a hyperparameter study.

This script extracts metrics from phase directories that use status.json files
(as created by run_full_study_parallel_180.sh) and writes them to CSV/Parquet
files for analysis.

Usage:
    python extract_phase1a_metrics.py [STUDY_DIR]
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Add src to path for imports
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def extract_metrics_from_logs(run_dir: Path) -> Dict[str, float]:
    """Extract metrics from training logs and WandB files.
    
    Enhanced version that extracts all metrics including latent metrics,
    cosine similarity, regularization loss, etc. from:
    1. metrics.json (if exists) - basic metrics from status.json
    2. WandB summary files (most comprehensive) - includes latent/cosine_similarity,
       latent/pred_loss, latent/reg_loss, and all other training metrics
    3. Training log files (stdout.log, stderr.log) - fallback for metrics not in WandB
    
    This function ensures we capture all metrics logged during training, not just
    the basic ones stored in status.json.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Dictionary of metric names to values, including:
        - latent/cosine_similarity
        - latent/pred_loss
        - latent/reg_loss
        - latent/pred_std_mean, latent/pred_std_min, latent/pred_std_max
        - latent/teacher_std_mean, latent/teacher_std_min, latent/teacher_std_max
        - latent/pred_cov_offdiag
        - latent/pred_pairwise_dist
        - latent/weight_mean
        - latent/num_masked
        - val/nll, train/nll
        - And more
    """
    metrics = {}
    
    # Try to load from metrics.json first (if training wrote it)
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            # Don't return early - we want to supplement with WandB and log parsing
        except (json.JSONDecodeError, IOError):
            pass
    
    # Try to load from WandB summary (most comprehensive source)
    wandb_dir = run_dir / "wandb"
    if wandb_dir.exists():
        # Find the BEST (most complete) wandb summary file
        # Some runs may have multiple wandb runs (retries, failures), so we want
        # the one with the most metrics
        best_summary = None
        best_metric_count = 0
        
        for run_subdir in wandb_dir.iterdir():
            if run_subdir.is_dir():
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
                        # Prefer summary with more metrics
                        if metric_count > best_metric_count:
                            best_summary = wandb_summary
                            best_metric_count = metric_count
                    except (json.JSONDecodeError, IOError):
                        continue
        
        # Merge best WandB metrics, prioritizing WandB for latent metrics
        if best_summary:
            for key, value in best_summary.items():
                # Skip internal WandB keys
                if key.startswith("_") or key.startswith("trainer/"):
                    continue
                # Convert to float if possible
                try:
                    if isinstance(value, (int, float)):
                        metrics[key] = float(value)
                    elif isinstance(value, bool):
                        metrics[key] = float(value)
                except (ValueError, TypeError):
                    pass
    
    # Parse from logs
    for log_file in ["stdout.log", "stderr.log"]:
        log_path = run_dir / log_file
        if not log_path.exists():
            # Also check parent logs directory
            config_idx = run_dir.name.split('_')[0] if '_' in run_dir.name else None
            if config_idx:
                parent_log = run_dir.parent.parent / "logs" / f"config_{config_idx}.out"
                if parent_log.exists():
                    log_path = parent_log
                else:
                    continue
            else:
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
                    # Keep the latest (best) value for metrics we want to minimize
                    # Keep the latest (best) value for metrics we want to maximize
                    if metric_name not in metrics:
                        metrics[metric_name] = metric_val
                    elif 'nll' in metric_name and metric_val < metrics[metric_name]:
                        metrics[metric_name] = metric_val
                    elif ('cosine' in metric_name or 'accuracy' in metric_name) and metric_val > metrics[metric_name]:
                        metrics[metric_name] = metric_val
                    elif 'nll' not in metric_name and 'cosine' not in metric_name and 'accuracy' not in metric_name:
                        # For other metrics, keep the last value
                        metrics[metric_name] = metric_val
                except ValueError:
                    continue
            
            # Pattern 2: Simple key=value format
            # val/nll=0.7241 or val_nll: 0.7241 or latent/cosine_similarity: 0.8567
            pattern2 = r"(val[/_][a-z_/]+|train[/_][a-z_/]+|latent[/_][a-z_/]+)[=:]\s*([\d.eE+-]+)"
            for match in re.finditer(pattern2, content):
                metric_name = match.group(1).replace('_', '/')
                try:
                    metric_val = float(match.group(2))
                    if metric_name not in metrics:
                        metrics[metric_name] = metric_val
                except ValueError:
                    continue
            
            # Pattern 3: WandB summary log format
            pattern3 = r'"(val/[^"]+|train/[^"]+|latent/[^"]+)":\s*([\d.eE+-]+)'
            for match in re.finditer(pattern3, content):
                metric_name = match.group(1)
                try:
                    metric_val = float(match.group(2))
                    if metric_name not in metrics:
                        metrics[metric_name] = metric_val
                except ValueError:
                    continue
            
            # Pattern 4: Lightning log format with step info
            # global step 5000: 'latent/cosine_similarity' = 0.8567
            pattern4 = r"global step \d+:\s*'(val/[^']+|train/[^']+|latent/[^']+)'\s*=\s*([\d.eE+-]+)"
            for match in re.finditer(pattern4, content):
                metric_name = match.group(1)
                try:
                    metric_val = float(match.group(2))
                    if metric_name not in metrics:
                        metrics[metric_name] = metric_val
                except ValueError:
                    continue
                    
        except (IOError, UnicodeDecodeError) as e:
            continue
    
    return metrics


def load_phase_results_from_status_json(phase_dir: Union[str, Path]) -> pd.DataFrame:
    """Load results from phase directory using status.json files.
    
    This function works with the storage format used by run_full_study_parallel_180.sh,
    where each run has a status.json file with metrics. It also extracts additional
    metrics from log files to capture all training metrics.
    
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
        }
        
        # Add config parameters
        config = cfg_data.get("config", {})
        for key, value in config.items():
            row[f"config.{key}"] = value
        
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
            
            # Add status information
            row["status"] = "completed" if status.get("exit_code", 1) == 0 else "failed"
            row["exit_code"] = status.get("exit_code", 1)
            row["duration_seconds"] = status.get("duration_seconds")
            row["hostname"] = status.get("hostname")
            row["gpus_per_config"] = status.get("gpus_per_config")
            row["checkpoint_path"] = status.get("checkpoint_path")
            row["started_at"] = status.get("started_at")
            row["completed_at"] = status.get("completed_at")
            
            # Get metrics from status.json
            metrics = status.get("metrics", {}).copy()
            
            # Extract additional metrics from logs if run completed successfully
            if row["status"] == "completed":
                log_metrics = extract_metrics_from_logs(run_dir)
                # Merge log metrics, preferring log metrics for latent metrics
                for key, value in log_metrics.items():
                    # Prioritize log-extracted metrics for latent/* metrics
                    if key.startswith("latent/") or key not in metrics:
                        metrics[key] = value
                    elif key not in metrics:
                        metrics[key] = value
            
            # Add all metrics to row
            for key, value in metrics.items():
                row[f"metric.{key}"] = value
        else:
            # Missing run
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
    
    # Phase directories to process
    phase_dirs = [
        study_dir / "phase1a_encoder",
        study_dir / "phase1b_arch_validation",
        study_dir / "phase2a_predictor",
        study_dir / "phase2b_training",
        study_dir / "phase2c_ema",
        study_dir / "phase3_decoder",
    ]
    
    all_dfs = []
    for phase_dir in phase_dirs:
        if phase_dir.exists() and (phase_dir / "configs.json").exists():
            try:
                df = load_phase_results_from_status_json(phase_dir)
                all_dfs.append(df)
                print(f"Loaded {len(df)} runs from {phase_dir.name}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not load {phase_dir.name} (JSON error): {e}")
                continue
            except Exception as e:
                print(f"Warning: Could not load {phase_dir.name}: {e}")
                continue
    
    if not all_dfs:
        raise ValueError(f"No phase data found in {study_dir}")
    
    # Combine all phases
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def generate_phase_summary(phase_dir: Union[str, Path], output_dir: Optional[Path] = None) -> Dict:
    """Generate summary statistics for a single phase.
    
    Uses existing analysis functions where possible.
    
    Args:
        phase_dir: Path to phase directory
        output_dir: Optional directory to save summary
    
    Returns:
        Summary dictionary
    """
    try:
        from discrete_diffusion.tuning.analysis import compute_feature_importance, find_optimal_configs
    except ImportError:
        # Fallback if imports fail
        compute_feature_importance = None
        find_optimal_configs = None
    
    phase_dir = Path(phase_dir)
    df = load_phase_results_from_status_json(phase_dir)
    
    completed = df[df["status"] == "completed"]
    
    summary = {
        "phase": phase_dir.name,
        "total_runs": len(df),
        "completed": len(completed),
        "failed": len(df[df["status"] == "failed"]),
        "missing": len(df[df["status"] == "missing"]),
    }
    
    # Metric statistics
    metric_cols = [c for c in df.columns if c.startswith("metric.")]
    metric_stats = {}
    for col in metric_cols:
        values = completed[col].dropna()
        if len(values) > 0:
            metric_stats[col] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "count": len(values),
            }
    summary["metric_statistics"] = metric_stats
    
    # Feature importance for key metrics
    if len(completed) > 5 and compute_feature_importance is not None:
        key_metrics = ["metric.val/nll", "metric.latent/cosine_similarity"]
        for metric_col in key_metrics:
            if metric_col in completed.columns and completed[metric_col].notna().sum() > 5:
                try:
                    importance_df = compute_feature_importance(
                        completed, metric_col, method="correlation"
                    )
                    summary[f"feature_importance_{metric_col.replace('metric.', '')}"] = (
                        importance_df.head(10).to_dict(orient="records")
                    )
                except Exception as e:
                    print(f"Warning: Could not compute feature importance for {metric_col}: {e}")
    
    # Top configurations
    if "metric.val/nll" in completed.columns and find_optimal_configs is not None:
        try:
            top_configs = find_optimal_configs(
                completed, "metric.val/nll", goal="minimize", top_k=5
            )
            summary["top_configurations"] = top_configs
        except Exception as e:
            print(f"Warning: Could not find optimal configs: {e}")
    
    # Save if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / f"{phase_dir.name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary to {summary_file}")
    
    return summary


def get_workspace_output_dir(study_dir: Path) -> Path:
    """Get workspace output directory for a study directory.
    
    Maps project directory to workspace directory:
    /home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa/sweep_outputs/...
    -> /hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/sweep_outputs/...
    
    Args:
        study_dir: Path to study directory
    
    Returns:
        Workspace output directory path
    """
    study_dir_str = str(study_dir)
    
    # Check if study_dir is in project directory
    project_prefix = "/home/hk-project-p0023960/hgf_nhz3359/text-diffusion-jepa/sweep_outputs/"
    workspace_prefix = "/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/sweep_outputs/"
    
    if study_dir_str.startswith(project_prefix):
        # Map to workspace directory
        relative_path = study_dir_str[len(project_prefix):]
        workspace_study_dir = Path(workspace_prefix) / relative_path
        return workspace_study_dir / "extracted_metrics"
    elif study_dir_str.startswith(workspace_prefix):
        # Already in workspace, use as-is
        return study_dir / "extracted_metrics"
    else:
        # Unknown path, use study_dir/extracted_metrics
        return study_dir / "extracted_metrics"


def extract_all_metrics(
    study_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Extract all metrics from a study and save to files.
    
    Args:
        study_dir: Path to study directory
        output_dir: Directory to save extracted metrics 
                   (default: workspace directory if study_dir is in project, else study_dir/extracted_metrics)
    
    Returns:
        Combined DataFrame with all runs
    """
    study_dir = Path(study_dir)
    
    if output_dir is None:
        # Default to workspace directory if study_dir is in project directory
        output_dir = get_workspace_output_dir(study_dir)
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting metrics from: {study_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load all phases
    combined_df = load_all_phases(study_dir)
    
    # Save combined results
    combined_csv = output_dir / "all_phases_combined.csv"
    
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nSaved combined results:")
    print(f"  CSV: {combined_csv}")
    print(f"  Total runs: {len(combined_df)}")
    
    # Try to save parquet (may fail due to disk quota)
    try:
        combined_parquet = output_dir / "all_phases_combined.parquet"
        combined_df.to_parquet(combined_parquet, index=False)
        print(f"  Parquet: {combined_parquet}")
    except Exception as e:
        print(f"  Warning: Could not save Parquet file: {e}")
    
    # Save per-phase results
    phase_dirs = [
        study_dir / "phase1a_encoder",
        study_dir / "phase1b_arch_validation",
        study_dir / "phase2a_predictor",
        study_dir / "phase2b_training",
        study_dir / "phase2c_ema",
        study_dir / "phase3_decoder",
    ]
    
    for phase_dir in phase_dirs:
        if phase_dir.exists() and (phase_dir / "configs.json").exists():
            try:
                phase_df = load_phase_results_from_status_json(phase_dir)
                
                # Save phase-specific files
                phase_csv = output_dir / f"{phase_dir.name}.csv"
                
                phase_df.to_csv(phase_csv, index=False)
                
                # Try to save parquet (may fail due to disk quota)
                try:
                    phase_parquet = output_dir / f"{phase_dir.name}.parquet"
                    phase_df.to_parquet(phase_parquet, index=False)
                except Exception:
                    pass  # Skip parquet if it fails
                
                # Save only completed runs
                completed = phase_df[phase_df["status"] == "completed"]
                if len(completed) > 0:
                    completed_csv = output_dir / f"{phase_dir.name}_completed.csv"
                    completed.to_csv(completed_csv, index=False)
                
                # Generate summary
                generate_phase_summary(phase_dir, output_dir)
                
                print(f"\n{phase_dir.name}:")
                print(f"  Total: {len(phase_df)}, Completed: {len(completed)}")
                
            except Exception as e:
                print(f"Warning: Could not process {phase_dir.name}: {e}")
                continue
    
    # Generate overall study summary
    study_summary = {
        "study_dir": str(study_dir),
        "total_runs": len(combined_df),
        "completed": len(combined_df[combined_df["status"] == "completed"]),
        "failed": len(combined_df[combined_df["status"] == "failed"]),
        "missing": len(combined_df[combined_df["status"] == "missing"]),
        "phases": {},
    }
    
    # Per-phase statistics
    for phase in combined_df["phase"].unique():
        phase_data = combined_df[combined_df["phase"] == phase]
        study_summary["phases"][phase] = {
            "total": len(phase_data),
            "completed": len(phase_data[phase_data["status"] == "completed"]),
            "failed": len(phase_data[phase_data["status"] == "failed"]),
        }
    
    summary_file = output_dir / "study_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(study_summary, f, indent=2, default=str)
    
    print(f"\n{'=' * 60}")
    print("Extraction complete!")
    print(f"Study summary: {summary_file}")
    
    return combined_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract metrics from hyperparameter study phases"
    )
    parser.add_argument(
        "study_dir",
        nargs="?",
        default="/hkfs/work/workspace/scratch/hgf_nhz3359-JEDi/sweep_outputs/jepa_180m_full_study_20251223_202845",
        help="Path to study directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: study_dir/extracted_metrics)"
    )
    
    args = parser.parse_args()
    
    # Extract metrics
    df = extract_all_metrics(args.study_dir, args.output_dir)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total runs: {len(df)}")
    print(f"Completed: {len(df[df['status'] == 'completed'])}")
    print(f"Failed: {len(df[df['status'] == 'failed'])}")
    print(f"Missing: {len(df[df['status'] == 'missing'])}")
    
    # Print metric columns
    metric_cols = [c for c in df.columns if c.startswith("metric.")]
    if metric_cols:
        print(f"\nAvailable metrics ({len(metric_cols)}):")
        for col in sorted(metric_cols)[:10]:
            print(f"  {col}")
        if len(metric_cols) > 10:
            print(f"  ... and {len(metric_cols) - 10} more")
