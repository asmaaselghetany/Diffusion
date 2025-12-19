"""Hyperparameter tuning module for Latent JEPA discrete diffusion.

This module provides systematic hyperparameter optimization following
the research plan in hyperparameter_tuning_plan.md.

Components:
- sweep_config: Parse and validate sweep configurations
- sweep_runner: Execute grid/random/latin_hypercube sweeps
- experiment_tracker: Track and aggregate experiment results
- early_stopping: Collapse detection and early termination
- analysis: Hyperparameter sensitivity analysis

Example usage:
    from discrete_diffusion.tuning import SweepRunner
    
    runner = SweepRunner('configs/sweep/stage1_180m_encoder.yaml')
    runner.run_sweep(num_runs=60)
"""

# Lazy imports to avoid circular dependencies and missing optional deps
def __getattr__(name):
    """Lazy import of module components."""
    if name == "SweepConfig":
        from .sweep_config import SweepConfig
        return SweepConfig
    elif name == "load_sweep_config":
        from .sweep_config import load_sweep_config
        return load_sweep_config
    elif name == "SweepRunner":
        from .sweep_runner import SweepRunner
        return SweepRunner
    elif name == "ExperimentTracker":
        from .experiment_tracker import ExperimentTracker
        return ExperimentTracker
    elif name == "CollapseDetectionCallback":
        from .early_stopping import CollapseDetectionCallback
        return CollapseDetectionCallback
    elif name == "PerformanceEarlyStoppingCallback":
        from .early_stopping import PerformanceEarlyStoppingCallback
        return PerformanceEarlyStoppingCallback
    elif name == "MetricLoggingCallback":
        from .early_stopping import MetricLoggingCallback
        return MetricLoggingCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'SweepConfig',
    'load_sweep_config',
    'SweepRunner',
    'ExperimentTracker',
    'CollapseDetectionCallback',
    'PerformanceEarlyStoppingCallback',
    'MetricLoggingCallback',
]
