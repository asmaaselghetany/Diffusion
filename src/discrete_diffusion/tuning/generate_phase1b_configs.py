#!/usr/bin/env python3
"""Generate Phase 1B (Architecture Validation) configurations.

This script extracts the top-k encoder configurations from Phase 1A
and top-k predictor configurations from Phase 2A, then generates
all k×k combinations for validation.

The goal is to validate that independently-found optimal encoder and
predictor architectures actually work well together, since they were
originally trained with baseline counterparts.

Usage:
    python -m discrete_diffusion.tuning.generate_phase1b_configs \
        --study-dir /path/to/study \
        --top-k 3 \
        --output-file /path/to/phase1b/configs.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Encoder parameters that vary in Phase 1A
ENCODER_PARAMS = [
    'model.latent_dim',
    'model.hidden_size',
    'model.n_heads',
    'model.n_blocks',
]

# Predictor parameters that vary in Phase 2A
PREDICTOR_PARAMS = [
    'model.predictor_depth',
    'model.predictor_hidden_size',
    'model.predictor_n_heads',
    'model.predictor_use_projections',
]

# Training parameters that vary in Phase 2B
# These are extracted from best Phase 2B run if available
TRAINING_PARAMS = [
    'algo.loss_norm',
    'algo.mask_only',
    'algo.normalize_targets',
    'algo.redundancy',
    'algo.lambda_var',
    'algo.lambda_cov',
]


def load_phase_results(phase_dir: Path) -> List[Dict[str, Any]]:
    """Load all successful run results from a phase directory.

    Reads metrics from wandb-summary.json (where latent/pred_loss and
    latent/reg_loss are stored) and combines with config from status.json.
    """
    results = []
    runs_dir = phase_dir / 'runs'

    if not runs_dir.exists():
        return results

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Check status.json for exit code
        status_file = run_dir / 'status.json'
        if not status_file.exists():
            continue

        try:
            with open(status_file) as f:
                status = json.load(f)

            if status.get('exit_code', 1) != 0:
                continue
        except (json.JSONDecodeError, IOError):
            continue

        # Find wandb-summary.json for metrics
        wandb_summary = None
        wandb_dir = run_dir / 'wandb'
        if wandb_dir.exists():
            for d in wandb_dir.iterdir():
                if d.is_dir() and d.name.startswith('run-'):
                    candidate = d / 'files' / 'wandb-summary.json'
                    if candidate.exists():
                        wandb_summary = candidate
                        break

        if not wandb_summary:
            continue

        try:
            with open(wandb_summary) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        pred_loss = metrics.get('latent/pred_loss')
        reg_loss = metrics.get('latent/reg_loss')

        if pred_loss is None or reg_loss is None:
            continue

        # Load config from config.yaml in run directory
        config_file = run_dir / 'config.yaml'
        config = {}
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    # Config is stored as key=value lines, not full YAML
                    content = f.read()
                    for line in content.strip().split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            # Try to parse value
                            try:
                                if value.lower() == 'true':
                                    value = True
                                elif value.lower() == 'false':
                                    value = False
                                elif '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                pass
                            config[key] = value
            except Exception:
                pass

        # Calculate combined metric: 20*reg_loss + pred_loss
        combined = 20.0 * reg_loss + pred_loss

        # Check for checkpoint
        ckpt = run_dir / 'checkpoints' / 'last.ckpt'
        if not ckpt.exists():
            ckpt = run_dir / 'checkpoints' / 'best.ckpt'

        results.append({
            'run_name': run_dir.name,
            'run_dir': str(run_dir),
            'config': config,
            'pred_loss': pred_loss,
            'reg_loss': reg_loss,
            'combined_metric': combined,
            'checkpoint': str(ckpt) if ckpt.exists() else None,
        })

    return results


def extract_top_k(
    results: List[Dict[str, Any]],
    param_keys: List[str],
    k: int
) -> List[Dict[str, Any]]:
    """Extract top-k results by combined metric, keeping only specified params."""
    # Sort by combined metric (lower is better)
    sorted_results = sorted(results, key=lambda x: x['combined_metric'])

    top_k = []
    seen_configs = set()

    for result in sorted_results:
        # Extract only the relevant parameters
        param_values = {}
        for key in param_keys:
            if key in result['config']:
                param_values[key] = result['config'][key]

        # Create a hashable key to avoid duplicates
        config_key = tuple(sorted(param_values.items()))

        if config_key not in seen_configs:
            seen_configs.add(config_key)
            top_k.append({
                'params': param_values,
                'source_run': result['run_name'],
                'combined_metric': result['combined_metric'],
                'pred_loss': result['pred_loss'],
                'reg_loss': result['reg_loss'],
            })

        if len(top_k) >= k:
            break

    return top_k


def extract_best_training_params(study_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract best training parameters from Phase 2B if it has completed runs.

    Returns None if Phase 2B doesn't exist or has no completed runs,
    allowing the caller to fall back to baseline parameters.

    Args:
        study_dir: Path to the study directory

    Returns:
        Dictionary of training parameters from best Phase 2B run, or None
    """
    phase2b_dir = study_dir / 'phase2b_training'
    results = load_phase_results(phase2b_dir)

    if not results:
        return None

    # Sort by combined metric (lower is better)
    sorted_results = sorted(results, key=lambda x: x['combined_metric'])
    best_run = sorted_results[0]

    # Extract training parameters from best run's config
    training_params = {}
    for key in TRAINING_PARAMS:
        if key in best_run['config']:
            training_params[key] = best_run['config'][key]

    if not training_params:
        return None

    # Include metadata about source
    return {
        'params': training_params,
        'source_run': best_run['run_name'],
        'combined_metric': best_run['combined_metric'],
        'pred_loss': best_run['pred_loss'],
        'reg_loss': best_run['reg_loss'],
    }


def get_baseline_config(model_size: str) -> Dict[str, Any]:
    """Get baseline fixed parameters for the model size."""
    if model_size == '180m':
        return {
            'algo': 'jepa',
            'algo.stage': 1,
            'model': 'latent_jepa_180M',
            'model.gradient_checkpointing': True,
            'model.dropout': 0.0,
            'model.time_embed_dim': 256,
            'model.readout_depth': 2,
            'model.readout_hidden_size': 512,
            'model.readout_type': 'tiny_transformer',
            'data': 'openwebtext-split',
            'data.cache_dir': '/home/hk-project-p0023960/hgf_nhz3359/New_Discrete_Diffusion-main/datasets/pgm_owt',
            'trainer.max_steps': 200000,
            'trainer.val_check_interval': 5000,
            'loader.global_batch_size': 256,
            'optim.lr': 3e-4,
        }
    elif model_size == '350m':
        return {
            'algo': 'jepa',
            'algo.stage': 1,
            'model': 'latent_jepa_350',
            'model.gradient_checkpointing': True,
            'model.dropout': 0.0,
            'model.time_embed_dim': 128,
            'model.readout_depth': 4,
            'model.readout_hidden_size': 768,
            'model.readout_type': 'transformer',
            'data': 'openwebtext-split',
            'data.cache_dir': '/home/hk-project-p0023960/hgf_nhz3359/New_Discrete_Diffusion-main/datasets/pgm_owt',
            'trainer.max_steps': 200000,
            'trainer.val_check_interval': 5000,
            'loader.global_batch_size': 256,
            'optim.lr': 3e-4,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def generate_phase1b_configs(
    study_dir: Path,
    top_k: int = 3,
    model_size: str = '180m',
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate Phase 1B configurations combining top encoders and predictors.

    If Phase 2B (training hyperparameters) has completed runs, the best
    training parameters from Phase 2B are used instead of baseline values.
    This ensures Phase 1B results are directly comparable with Phase 2B
    results when Phase 2B has already run (e.g., in resume scenarios).

    Args:
        study_dir: Path to the study directory
        top_k: Number of top configurations to extract from each phase
        model_size: Model size ('180m' or '350m')

    Returns:
        Tuple of (configs list, metadata dict)
    """
    phase1a_dir = study_dir / 'phase1a_encoder'
    phase2a_dir = study_dir / 'phase2a_predictor'

    # Load results from both phases
    encoder_results = load_phase_results(phase1a_dir)
    predictor_results = load_phase_results(phase2a_dir)

    if not encoder_results:
        raise ValueError(f"No successful encoder runs found in {phase1a_dir}")
    if not predictor_results:
        raise ValueError(f"No successful predictor runs found in {phase2a_dir}")

    # Extract top-k from each
    top_encoders = extract_top_k(encoder_results, ENCODER_PARAMS, top_k)
    top_predictors = extract_top_k(predictor_results, PREDICTOR_PARAMS, top_k)

    # Get baseline config
    baseline = get_baseline_config(model_size)

    # Check if Phase 2B has completed runs - if so, use best training params
    # This enables better comparison when Phase 2B has already run
    training_params_info = extract_best_training_params(study_dir)
    training_params_source = 'baseline'

    if training_params_info is not None:
        # Apply best training params from Phase 2B to baseline
        baseline.update(training_params_info['params'])
        training_params_source = 'phase2b'
        print(f"Using training params from Phase 2B best run: {training_params_info['source_run']}")
        print(f"  Training params: {training_params_info['params']}")
    else:
        print("Phase 2B not complete - using baseline training parameters")

    # Generate all k×k combinations
    configs = []
    for enc_idx, encoder in enumerate(top_encoders):
        for pred_idx, predictor in enumerate(top_predictors):
            config = baseline.copy()

            # Add encoder params
            config.update(encoder['params'])

            # Add predictor params
            config.update(predictor['params'])

            configs.append(config)

    # Prepare metadata for logging
    metadata = {
        'top_encoders': top_encoders,
        'top_predictors': top_predictors,
        'num_encoder_candidates': len(encoder_results),
        'num_predictor_candidates': len(predictor_results),
        'top_k': top_k,
        'total_combinations': len(configs),
        # Track training params source for transparency
        'training_params_source': training_params_source,
        'training_params_info': training_params_info,
    }

    return configs, metadata


def get_run_name(config: Dict[str, Any], idx: int) -> str:
    """Generate a descriptive run name for a Phase 1B config."""
    enc_parts = []
    pred_parts = []

    for key, value in config.items():
        short_key = key.split('.')[-1]
        if key in ENCODER_PARAMS:
            enc_parts.append(f"{short_key}_{value}")
        elif key in PREDICTOR_PARAMS:
            pred_parts.append(f"{short_key}_{value}")

    enc_str = '_'.join(enc_parts[:2])  # Limit length
    pred_str = '_'.join(pred_parts[:2])

    return f"phase1b_arch_validation_{idx:04d}_enc_{enc_str}_pred_{pred_str}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 1B architecture validation configs"
    )
    parser.add_argument(
        '--study-dir', type=str, required=True,
        help='Path to the study directory'
    )
    parser.add_argument(
        '--top-k', type=int, default=3,
        help='Number of top configurations to extract from each phase'
    )
    parser.add_argument(
        '--model-size', type=str, default='180m',
        choices=['180m', '350m'],
        help='Model size'
    )
    parser.add_argument(
        '--output-file', type=str, required=True,
        help='Output path for configs.json'
    )

    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    output_file = Path(args.output_file)

    print(f"Generating Phase 1B configs for study: {study_dir}")
    print(f"Top-k: {args.top_k}, Model size: {args.model_size}")

    configs, metadata = generate_phase1b_configs(
        study_dir=study_dir,
        top_k=args.top_k,
        model_size=args.model_size,
    )

    # Format output with run names
    output = []
    for idx, config in enumerate(configs):
        run_name = get_run_name(config, idx)
        output.append({
            'idx': idx,
            'run_name': run_name,
            'config': config,
        })

    # Save configs
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save metadata
    metadata_file = output_file.parent / 'phase1b_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nGenerated {len(output)} configurations")
    print(f"Configs saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")

    print("\nTop Encoders:")
    for i, enc in enumerate(metadata['top_encoders']):
        print(f"  {i+1}. {enc['source_run']}")
        print(f"     combined={enc['combined_metric']:.4f}, params={enc['params']}")

    print("\nTop Predictors:")
    for i, pred in enumerate(metadata['top_predictors']):
        print(f"  {i+1}. {pred['source_run']}")
        print(f"     combined={pred['combined_metric']:.4f}, params={pred['params']}")

    # Print training params source
    print(f"\nTraining Parameters Source: {metadata['training_params_source']}")
    if metadata['training_params_info'] is not None:
        info = metadata['training_params_info']
        print(f"  Source run: {info['source_run']}")
        print(f"  Combined metric: {info['combined_metric']:.4f}")
        print(f"  Parameters applied:")
        for key, value in info['params'].items():
            print(f"    {key}: {value}")
    else:
        print("  Using baseline training parameters (Phase 2B not complete)")


if __name__ == '__main__':
    main()
