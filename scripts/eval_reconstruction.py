#!/usr/bin/env python3
"""Evaluate reconstruction quality of Stage 2 decoder checkpoints.

This script:
1. Loads each best.ckpt from the Stage 2 decoder runs
2. Feeds clean sequences through the model: encode → predict → decode
3. Computes reconstruction metrics (accuracy, token match rate)
4. Outputs results as JSON

Usage:
    python scripts/eval_reconstruction.py \
        --checkpoint_dir /path/to/checkpoints \
        --num_samples 10 \
        --output_path reconstruction_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra.utils
from discrete_diffusion.data import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction quality of Stage 2 checkpoints"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing Stage 2 checkpoint subdirectories",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate per checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="reconstruction_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save original and reconstructed text samples alongside metrics",
    )
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default="*/checkpoints/best.ckpt",
        help="Glob pattern to find checkpoints within checkpoint_dir",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def find_checkpoints(checkpoint_dir: str, pattern: str) -> List[Path]:
    """Find all checkpoints matching the pattern."""
    search_path = str(Path(checkpoint_dir) / pattern)
    paths = sorted(glob(search_path))
    return [Path(p) for p in paths]


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load a LatentJEPATrainer model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model onto
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "hyper_parameters" not in ckpt:
        raise ValueError("Checkpoint does not contain 'hyper_parameters'")
    if "config" not in ckpt["hyper_parameters"]:
        raise ValueError("Checkpoint hyper_parameters does not contain 'config'")
    
    model_config = ckpt["hyper_parameters"]["config"]
    
    # Ensure it's an OmegaConf object
    if not OmegaConf.is_config(model_config):
        model_config = OmegaConf.create(model_config)
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_config)
    
    # Identify algorithm class
    algo_target = model_config.algo._target_
    algo_cls = hydra.utils.get_class(algo_target)
    print(f"  Algorithm class: {algo_cls.__name__}")
    
    # Load model from checkpoint
    model = algo_cls.load_from_checkpoint(
        str(checkpoint_path),
        config=model_config,
        tokenizer=tokenizer,
        map_location=device,
        strict=False,
    )
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, model_config


def get_sample_sequences(tokenizer, num_samples: int, seq_length: int, seed: int) -> List[str]:
    """Get sample text sequences for reconstruction testing.
    
    Uses a fixed set of diverse text samples for reproducibility.
    """
    # Fixed diverse sample texts for reconstruction testing
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once.",
        "In the beginning, there was nothing but darkness and chaos. Then, a spark of light emerged from the void.",
        "Machine learning has revolutionized the way we process and understand data. Neural networks can now perform tasks that were once thought impossible.",
        "The ancient library of Alexandria was one of the largest and most significant libraries of the ancient world. It was dedicated to the Muses.",
        "Climate change poses one of the greatest challenges facing humanity today. Rising temperatures and extreme weather events threaten ecosystems worldwide.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses.",
        "Music has been an integral part of human culture for thousands of years. From ancient drums to modern synthesizers, it continues to evolve.",
        "The development of the printing press by Johannes Gutenberg in the 15th century revolutionized the spread of knowledge across Europe.",
        "Quantum mechanics describes the behavior of matter and energy at the smallest scales. Particles can exist in superposition states.",
        "The Renaissance was a period of cultural, artistic, and intellectual rebirth that began in Italy during the 14th century.",
        "Artificial intelligence systems are now capable of generating human-like text, images, and even music with remarkable quality.",
        "The Great Wall of China stretches over 13,000 miles and was built over many centuries to protect against invasions from the north.",
        "DNA carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms.",
        "The Industrial Revolution transformed society from agrarian economies to industrial powerhouses, changing the course of history.",
        "Space exploration has expanded our understanding of the universe, from the first moon landing to the exploration of Mars.",
    ]
    
    # Use seed for reproducible selection
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(sample_texts), size=min(num_samples, len(sample_texts)), replace=False)
    
    # If we need more samples than available, repeat with different seeds
    selected_texts = []
    for i in range(num_samples):
        idx = selected_indices[i % len(selected_indices)]
        text = sample_texts[idx]
        # Add variation for repeated samples
        if i >= len(selected_indices):
            text = f"Sample {i}: " + text
        selected_texts.append(text)
    
    return selected_texts


def reconstruct_sequences(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct sequences through encode → predict → decode.
    
    Args:
        model: The loaded LatentJEPATrainer model
        tokenizer: Tokenizer for encoding/decoding
        input_ids: Input token IDs [batch_size, seq_length]
        device: Device to run on
        
    Returns:
        Tuple of (reconstructed_ids, logits)
    """
    backbone = model.backbone
    
    with torch.no_grad():
        # Use t=0 for clean reconstruction (no noise)
        t = torch.zeros(input_ids.shape[0], device=device, dtype=torch.float32)
        
        # Encode: tokens → latent
        z_t = backbone.encode_student(input_ids, t)
        
        # Predict: latent → predicted latent (at t=0, should be identity-like)
        z_hat_0 = backbone.predict_latent(z_t, t)
        
        # Decode: latent → logits
        logits = backbone.readout_tokens(z_hat_0)
        
        # Get reconstructed tokens (argmax)
        reconstructed_ids = logits.argmax(dim=-1)
    
    return reconstructed_ids, logits


def compute_reconstruction_metrics(
    original_ids: torch.Tensor,
    reconstructed_ids: torch.Tensor,
    tokenizer,
    pad_token_id: Optional[int] = None,
) -> dict:
    """Compute reconstruction metrics.
    
    Args:
        original_ids: Original token IDs [batch_size, seq_length]
        reconstructed_ids: Reconstructed token IDs [batch_size, seq_length]
        tokenizer: Tokenizer for decoding
        pad_token_id: ID of padding token to exclude from metrics
        
    Returns:
        Dict with reconstruction metrics
    """
    batch_size, seq_length = original_ids.shape
    
    # Token-level accuracy
    matches = (original_ids == reconstructed_ids)
    
    # Exclude padding tokens if specified
    if pad_token_id is not None:
        valid_mask = (original_ids != pad_token_id)
        valid_matches = matches & valid_mask
        total_valid = valid_mask.sum().item()
        total_matches = valid_matches.sum().item()
    else:
        total_valid = original_ids.numel()
        total_matches = matches.sum().item()
    
    token_accuracy = total_matches / total_valid if total_valid > 0 else 0.0
    
    # Per-sequence accuracy
    if pad_token_id is not None:
        seq_matches = (matches | ~valid_mask).all(dim=-1)
    else:
        seq_matches = matches.all(dim=-1)
    sequence_accuracy = seq_matches.float().mean().item()
    
    # Per-sequence token match rate
    per_seq_accuracy = []
    for i in range(batch_size):
        if pad_token_id is not None:
            valid = (original_ids[i] != pad_token_id)
            seq_match_rate = matches[i][valid].float().mean().item() if valid.any() else 1.0
        else:
            seq_match_rate = matches[i].float().mean().item()
        per_seq_accuracy.append(seq_match_rate)
    
    return {
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "mean_per_seq_accuracy": float(np.mean(per_seq_accuracy)),
        "std_per_seq_accuracy": float(np.std(per_seq_accuracy)),
        "total_tokens": total_valid,
        "matched_tokens": total_matches,
        "per_sequence_accuracy": per_seq_accuracy,
    }


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir, args.checkpoint_pattern)
    if not checkpoints:
        print(f"No checkpoints found at {args.checkpoint_dir}/{args.checkpoint_pattern}")
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.parent.parent.name}")
    
    # Process each checkpoint
    results = []
    
    for ckpt_path in tqdm(checkpoints, desc="Processing checkpoints"):
        run_name = ckpt_path.parent.parent.name
        print(f"\n{'='*60}")
        print(f"Processing: {run_name}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model, model_tokenizer, config = load_model_from_checkpoint(
                ckpt_path, device
            )
            
            # Get sequence length from config
            seq_length = config.model.length
            
            # Get sample texts
            sample_texts = get_sample_sequences(
                model_tokenizer,
                args.num_samples,
                seq_length,
                args.seed,
            )
            
            # Tokenize samples
            print(f"Tokenizing {len(sample_texts)} samples (seq_length={seq_length})...")
            encoded = model_tokenizer(
                sample_texts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=seq_length,
            )
            input_ids = encoded["input_ids"].to(device)
            
            # Reconstruct
            print("Reconstructing sequences...")
            reconstructed_ids, logits = reconstruct_sequences(
                model,
                model_tokenizer,
                input_ids,
                device,
            )
            
            # Compute metrics
            pad_token_id = model_tokenizer.pad_token_id
            metrics = compute_reconstruction_metrics(
                input_ids,
                reconstructed_ids,
                model_tokenizer,
                pad_token_id,
            )
            
            # Decode texts
            original_texts = model_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            reconstructed_texts = model_tokenizer.batch_decode(reconstructed_ids, skip_special_tokens=True)
            
            # Print sample preview
            print(f"\nSample reconstruction preview:")
            print(f"  Original:      {original_texts[0][:100]}...")
            print(f"  Reconstructed: {reconstructed_texts[0][:100]}...")
            
            result = {
                "checkpoint": run_name,
                "checkpoint_path": str(ckpt_path),
                "token_accuracy": metrics["token_accuracy"],
                "sequence_accuracy": metrics["sequence_accuracy"],
                "mean_per_seq_accuracy": metrics["mean_per_seq_accuracy"],
                "std_per_seq_accuracy": metrics["std_per_seq_accuracy"],
                "total_tokens": metrics["total_tokens"],
                "matched_tokens": metrics["matched_tokens"],
            }
            
            if args.save_samples:
                result["samples"] = [
                    {
                        "original": orig,
                        "reconstructed": recon,
                        "accuracy": acc,
                    }
                    for orig, recon, acc in zip(
                        original_texts, reconstructed_texts, metrics["per_sequence_accuracy"]
                    )
                ]
            
            results.append(result)
            
            print(f"\nResults for {run_name}:")
            print(f"  Token Accuracy: {metrics['token_accuracy']:.4f} ({metrics['token_accuracy']*100:.2f}%)")
            print(f"  Sequence Accuracy: {metrics['sequence_accuracy']:.4f} ({metrics['sequence_accuracy']*100:.2f}%)")
            print(f"  Mean Per-Seq Accuracy: {metrics['mean_per_seq_accuracy']:.4f} ± {metrics['std_per_seq_accuracy']:.4f}")
            
            # Free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {run_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "checkpoint": run_name,
                "checkpoint_path": str(ckpt_path),
                "error": str(e),
            })
    
    # Compile final output
    output = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\nSummary:")
    print("-" * 90)
    print(f"{'Checkpoint':<50} {'Token Acc':>12} {'Seq Acc':>12} {'Mean±Std':>15}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['checkpoint']:<50} {'ERROR':>12}")
        else:
            mean_std = f"{r['mean_per_seq_accuracy']:.4f}±{r['std_per_seq_accuracy']:.4f}"
            print(f"{r['checkpoint']:<50} {r['token_accuracy']:>11.4f} {r['sequence_accuracy']:>11.4f} {mean_std:>15}")
    print("-" * 90)


if __name__ == "__main__":
    main()


