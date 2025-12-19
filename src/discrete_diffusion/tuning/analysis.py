"""Analysis utilities for hyperparameter sensitivity studies.

Provides tools for analyzing sweep results, computing feature importance,
generating visualizations, and identifying optimal configurations.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def load_sweep_results(sweep_dir: Union[str, Path]) -> pd.DataFrame:
    """Load results from a sweep directory.
    
    Args:
        sweep_dir: Path to sweep output directory
    
    Returns:
        DataFrame with all runs and metrics
    """
    from .experiment_tracker import ExperimentTracker
    
    sweep_dir = Path(sweep_dir)
    db_path = sweep_dir / "experiments.json"
    
    if not db_path.exists():
        raise FileNotFoundError(f"No experiments.json found in {sweep_dir}")
    
    tracker = ExperimentTracker(db_path)
    return tracker.to_dataframe()


def compute_feature_importance(
    df: pd.DataFrame,
    target_metric: str,
    feature_prefix: str = "config.",
    method: str = "correlation"
) -> pd.DataFrame:
    """Compute feature importance for hyperparameters.
    
    Args:
        df: DataFrame with sweep results
        target_metric: Metric column to predict (e.g., 'metric.latent/cosine_similarity')
        feature_prefix: Prefix for feature columns
        method: 'correlation', 'mutual_info', or 'random_forest'
    
    Returns:
        DataFrame with feature importance scores
    """
    # Get feature columns
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    
    if not feature_cols:
        raise ValueError(f"No columns with prefix '{feature_prefix}'")
    
    if target_metric not in df.columns:
        raise ValueError(f"Target metric '{target_metric}' not in DataFrame")
    
    # Filter valid rows
    valid_mask = df[target_metric].notna()
    for col in feature_cols:
        valid_mask &= df[col].notna()
    
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) < 5:
        raise ValueError(f"Not enough valid samples ({len(df_valid)})")
    
    # Encode categorical features
    X = df_valid[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    y = df_valid[target_metric].values
    
    if method == "correlation":
        # Pearson correlation
        importances = []
        for col in X.columns:
            corr = np.corrcoef(X[col].values, y)[0, 1]
            importances.append({
                "feature": col.replace(feature_prefix, ""),
                "importance": abs(corr),
                "direction": "positive" if corr > 0 else "negative",
                "correlation": corr
            })
    
    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_regression
        
        mi_scores = mutual_info_regression(X.values, y)
        importances = [
            {"feature": col.replace(feature_prefix, ""), "importance": score}
            for col, score in zip(X.columns, mi_scores)
        ]
    
    elif method == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X.values, y)
        
        importances = [
            {"feature": col.replace(feature_prefix, ""), "importance": score}
            for col, score in zip(X.columns, rf.feature_importances_)
        ]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return pd.DataFrame(importances).sort_values("importance", ascending=False)


def compute_interaction_effects(
    df: pd.DataFrame,
    target_metric: str,
    param_pairs: Optional[List[Tuple[str, str]]] = None,
    feature_prefix: str = "config."
) -> pd.DataFrame:
    """Compute pairwise interaction effects between hyperparameters.
    
    Args:
        df: DataFrame with sweep results
        target_metric: Metric to analyze
        param_pairs: Specific pairs to analyze (or all if None)
        feature_prefix: Prefix for feature columns
    
    Returns:
        DataFrame with interaction effect strengths
    """
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    
    if param_pairs is None:
        # Generate all pairs
        param_pairs = [
            (feature_cols[i], feature_cols[j])
            for i in range(len(feature_cols))
            for j in range(i + 1, len(feature_cols))
        ]
    
    interactions = []
    
    for col1, col2 in param_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            continue
        
        # Group by both parameters
        try:
            grouped = df.groupby([col1, col2])[target_metric].agg(['mean', 'std', 'count'])
            
            if len(grouped) < 4:
                continue
            
            # Compute interaction strength as variance of group means
            group_means = grouped['mean'].values
            interaction_strength = np.std(group_means) if len(group_means) > 1 else 0
            
            interactions.append({
                "param1": col1.replace(feature_prefix, ""),
                "param2": col2.replace(feature_prefix, ""),
                "interaction_strength": interaction_strength,
                "num_combinations": len(grouped)
            })
        except Exception:
            continue
    
    return pd.DataFrame(interactions).sort_values("interaction_strength", ascending=False)


def find_optimal_configs(
    df: pd.DataFrame,
    target_metric: str,
    goal: str = "maximize",
    top_k: int = 5,
    constraints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Find optimal configurations from sweep results.
    
    Args:
        df: DataFrame with sweep results
        target_metric: Metric to optimize
        goal: 'maximize' or 'minimize'
        top_k: Number of top configurations to return
        constraints: Optional constraints on hyperparameters
    
    Returns:
        List of top-k configurations with their metrics
    """
    # Filter by constraints
    filtered = df.copy()
    if constraints:
        for key, value in constraints.items():
            if key in filtered.columns:
                filtered = filtered[filtered[key] == value]
    
    # Filter valid rows
    filtered = filtered[filtered[target_metric].notna()]
    
    if len(filtered) == 0:
        return []
    
    # Sort by metric
    ascending = goal == "minimize"
    sorted_df = filtered.sort_values(target_metric, ascending=ascending)
    
    # Extract top-k
    results = []
    config_cols = [c for c in sorted_df.columns if c.startswith("config.")]
    metric_cols = [c for c in sorted_df.columns if c.startswith("metric.")]
    
    for _, row in sorted_df.head(top_k).iterrows():
        config = {c.replace("config.", ""): row[c] for c in config_cols}
        metrics = {c.replace("metric.", ""): row[c] for c in metric_cols if pd.notna(row[c])}
        
        results.append({
            "config": config,
            "metrics": metrics,
            "run_id": row.get("run_id", "unknown")
        })
    
    return results


def generate_sweep_report(
    sweep_dir: Union[str, Path],
    target_metric: str,
    goal: str = "maximize",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Generate comprehensive report for a sweep.
    
    Args:
        sweep_dir: Path to sweep output directory
        target_metric: Primary metric to optimize
        goal: 'maximize' or 'minimize'
        output_path: Optional path to save report
    
    Returns:
        Report dictionary
    """
    df = load_sweep_results(sweep_dir)
    
    # Basic statistics
    total_runs = len(df)
    completed = len(df[df["status"] == "completed"])
    failed = len(df[df["status"].isin(["failed", "error"])])
    
    # Metric statistics
    metric_col = f"metric.{target_metric}" if not target_metric.startswith("metric.") else target_metric
    if metric_col in df.columns:
        metric_values = df[df["status"] == "completed"][metric_col].dropna()
        metric_stats = {
            "mean": float(metric_values.mean()),
            "std": float(metric_values.std()),
            "min": float(metric_values.min()),
            "max": float(metric_values.max()),
            "count": len(metric_values)
        }
    else:
        metric_stats = {}
    
    # Feature importance
    try:
        importance_df = compute_feature_importance(df, metric_col)
        feature_importance = importance_df.to_dict(orient="records")
    except Exception as e:
        feature_importance = []
        print(f"Warning: Could not compute feature importance: {e}")
    
    # Top configurations
    top_configs = find_optimal_configs(df, metric_col, goal=goal, top_k=5)
    
    # Build report
    report = {
        "sweep_dir": str(sweep_dir),
        "target_metric": target_metric,
        "goal": goal,
        "summary": {
            "total_runs": total_runs,
            "completed": completed,
            "failed": failed,
            "completion_rate": completed / total_runs if total_runs > 0 else 0
        },
        "metric_statistics": metric_stats,
        "feature_importance": feature_importance,
        "top_configurations": top_configs,
    }
    
    # Save if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {output_path}")
    
    return report


def compare_sweeps(
    sweep_dirs: List[Union[str, Path]],
    target_metric: str,
    goal: str = "maximize"
) -> pd.DataFrame:
    """Compare results across multiple sweeps.
    
    Args:
        sweep_dirs: List of sweep directories
        target_metric: Metric to compare
        goal: 'maximize' or 'minimize'
    
    Returns:
        DataFrame comparing sweep results
    """
    comparisons = []
    
    for sweep_dir in sweep_dirs:
        sweep_dir = Path(sweep_dir)
        try:
            df = load_sweep_results(sweep_dir)
            metric_col = f"metric.{target_metric}" if not target_metric.startswith("metric.") else target_metric
            
            completed = df[df["status"] == "completed"]
            if len(completed) == 0 or metric_col not in completed.columns:
                continue
            
            values = completed[metric_col].dropna()
            
            if goal == "maximize":
                best_idx = values.idxmax()
            else:
                best_idx = values.idxmin()
            
            comparisons.append({
                "sweep": sweep_dir.name,
                "num_completed": len(completed),
                "best_value": values.loc[best_idx],
                "mean_value": values.mean(),
                "std_value": values.std(),
                "best_run_id": completed.loc[best_idx, "run_id"] if "run_id" in completed.columns else None
            })
        except Exception as e:
            print(f"Warning: Could not load {sweep_dir}: {e}")
    
    return pd.DataFrame(comparisons).sort_values("best_value", ascending=(goal == "minimize"))


def main():
    """CLI for sweep analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument("sweep_dir", type=str, help="Sweep output directory")
    parser.add_argument("--metric", type=str, default="latent/cosine_similarity",
                       help="Target metric to analyze")
    parser.add_argument("--goal", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--output", type=str, default=None, help="Output path for report")
    parser.add_argument("--importance-method", type=str, default="correlation",
                       choices=["correlation", "mutual_info", "random_forest"])
    
    args = parser.parse_args()
    
    report = generate_sweep_report(
        args.sweep_dir,
        args.metric,
        args.goal,
        args.output or f"{args.sweep_dir}/report.json"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SWEEP ANALYSIS REPORT")
    print("="*60)
    print(f"\nSummary:")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    
    print(f"\n{args.metric} Statistics:")
    for k, v in report["metric_statistics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nTop 5 Feature Importances:")
    for i, feat in enumerate(report["feature_importance"][:5], 1):
        print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
    
    print(f"\nBest Configuration:")
    if report["top_configurations"]:
        best = report["top_configurations"][0]
        print(f"  Run ID: {best['run_id']}")
        print(f"  Metrics: {best['metrics']}")
        print(f"  Config: {best['config']}")


if __name__ == "__main__":
    main()

