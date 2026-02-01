#!/usr/bin/env python3
"""Evaluate MDLM checkpoints from New_Discrete_Diffusion-main with Generative Perplexity.

This script:
1. Loads MDLM checkpoints from New_Discrete_Diffusion-main
2. Generates sequences using the native AbsorbingSampler
3. Computes generative perplexity using GPT2-XL
4. Outputs results as JSON

Usage:
    python scripts/eval_mdlm_gen_ppl.py \
        --checkpoint_path /path/to/best.ckpt \
        --num_samples 10 \
        --num_steps 256 \
        --output_path gen_ppl_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add New_Discrete_Diffusion-main to path
NEW_DD_PATH = Path("/home/hk-project-p0023960/hgf_nhz3359/New_Discrete_Diffusion-main/src")
sys.path.insert(0, str(NEW_DD_PATH))

import hydra.utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MDLM checkpoints with Generative Perplexity"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the MDLM checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=256,
        help="Number of denoising steps for generation",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt2-xl",
        help="Pretrained model for perplexity evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation and evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max sequence length for evaluation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="gen_ppl_results.json",
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
        help="Save generated text samples alongside metrics",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="none",
        choices=["none", "prefix", "infill"],
        help="Prompt mode: 'none' (unconditional), 'prefix' (left-to-right completion), 'infill' (fill middle)",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="Prompt text for prefix mode",
    )
    parser.add_argument(
        "--suffix_text",
        type=str,
        default=None,
        help="Suffix text for infill mode",
    )
    parser.add_argument(
        "--generated_length",
        type=int,
        default=200,
        help="Number of tokens to generate (for prefix/infill modes)",
    )
    return parser.parse_args()


def load_mdlm_model(checkpoint_path: str, device: torch.device):
    """Load an MDLM model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model onto
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    print(f"Loading MDLM checkpoint from {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "hyper_parameters" not in ckpt:
        raise ValueError("Checkpoint does not contain 'hyper_parameters'")
    if "config" not in ckpt["hyper_parameters"]:
        raise ValueError("Checkpoint hyper_parameters does not contain 'config'")
    
    config = ckpt["hyper_parameters"]["config"]
    
    # Ensure it's an OmegaConf object
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    
    # Get tokenizer
    from discrete_diffusion.data import get_tokenizer
    tokenizer = get_tokenizer(config)
    
    # Identify algorithm class
    algo_target = config.algo._target_
    algo_cls = hydra.utils.get_class(algo_target)
    print(f"  Algorithm class: {algo_cls.__name__}")
    
    # Load model from checkpoint
    model = algo_cls.load_from_checkpoint(
        str(checkpoint_path),
        config=config,
        tokenizer=tokenizer,
        map_location=device,
        strict=False,
    )
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, config


def generate_samples_unconditional(
    model,
    num_samples: int,
    num_steps: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Generate unconditional samples from the MDLM model.
    
    Args:
        model: The loaded MDLM model
        num_samples: Total number of samples to generate
        num_steps: Number of denoising steps
        batch_size: Batch size for generation
        
    Returns:
        Tensor of token IDs [num_samples, seq_length]
    """
    all_samples = []
    remaining = num_samples
    
    with torch.no_grad():
        while remaining > 0:
            curr_batch = min(batch_size, remaining)
            samples = model.generate_samples(
                num_samples=curr_batch,
                num_steps=num_steps,
            )
            all_samples.append(samples.detach().cpu())
            remaining -= curr_batch
    
    return torch.cat(all_samples, dim=0)


def generate_samples_with_prompt(
    model,
    tokenizer,
    num_samples: int,
    num_steps: int,
    prompt_text: str,
    suffix_text: Optional[str] = None,
    generated_length: int = 200,
    batch_size: int = 8,
) -> torch.Tensor:
    """Generate samples with a prompt (prefix) and optionally a suffix (infill).
    
    Args:
        model: The loaded MDLM model
        tokenizer: Tokenizer for the model
        num_samples: Total number of samples to generate
        num_steps: Number of denoising steps
        prompt_text: Prompt text for prefix/infill modes
        suffix_text: Suffix text for infill mode (optional)
        generated_length: Number of tokens to generate
        batch_size: Batch size for generation
        
    Returns:
        Tensor of token IDs [num_samples, total_length]
    """
    device = model.device
    mask_id = model.mask_id
    
    # Tokenize prompt
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt_text, add_special_tokens=False),
        dtype=torch.long,
        device=device
    )
    prompt_len = len(prompt_ids)
    
    # Tokenize suffix if provided
    suffix_ids = None
    suffix_len = 0
    if suffix_text:
        suffix_ids = torch.tensor(
            tokenizer.encode(suffix_text, add_special_tokens=False),
            dtype=torch.long,
            device=device
        )
        suffix_len = len(suffix_ids)
    
    total_length = min(prompt_len + generated_length + suffix_len, model.num_tokens)
    
    all_samples = []
    remaining = num_samples
    
    with torch.no_grad():
        while remaining > 0:
            curr_batch = min(batch_size, remaining)
            
            # Initialize with all masks
            x = torch.full((curr_batch, total_length), mask_id, dtype=torch.long, device=device)
            
            # Set prompt at the beginning
            x[:, :prompt_len] = prompt_ids.unsqueeze(0).expand(curr_batch, -1)
            
            # Set suffix at the end if provided
            if suffix_ids is not None and suffix_len > 0:
                suffix_start = total_length - suffix_len
                x[:, suffix_start:] = suffix_ids.unsqueeze(0).expand(curr_batch, -1)
            
            # Create fixed mask (positions that should NOT be denoised)
            fix_mask = torch.zeros_like(x, dtype=torch.bool)
            fix_mask[:, :prompt_len] = True
            if suffix_ids is not None and suffix_len > 0:
                fix_mask[:, suffix_start:] = True
            
            # Run denoising loop
            eps = 1e-5
            timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
            dt = (1 - eps) / num_steps
            
            for i in range(num_steps):
                t = timesteps[i] * torch.ones(curr_batch, 1, device=device)
                
                # Get predictions
                alpha_t = model.noise.alpha_t(t)
                alpha_s = model.noise.alpha_t(t - dt) if i < num_steps - 1 else torch.ones_like(alpha_t)
                
                sigma = -torch.log(alpha_t)
                log_p_x0 = model.forward(x, sigma)
                p_x0 = log_p_x0.exp()
                
                # Sample from predictions
                from discrete_diffusion.forward_process.utils import sample_categorical
                sampled_x0 = sample_categorical(p_x0)
                
                # Decide which positions to denoise
                prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t)
                should_denoise_draw = torch.rand_like(x, dtype=torch.float64, device=device) < prob_denoise
                is_masked = (x == mask_id)
                should_denoise_mask = is_masked & should_denoise_draw & (~fix_mask)
                
                # Update x
                _x = torch.where(should_denoise_mask, sampled_x0, x)
                x = torch.where(x != mask_id, x, _x)
            
            # Final denoising step
            t0 = timesteps[-1] * torch.ones(curr_batch, 1, device=device)
            alpha_t0 = model.noise.alpha_t(t0)
            sigma_t0 = -torch.log(alpha_t0)
            log_p_x0 = model.forward(x, sigma_t0)
            p_x0 = log_p_x0.exp()
            sampled_x0 = sample_categorical(p_x0)
            
            # Replace remaining masks (respecting fix_mask)
            remaining_masks = (x == mask_id) & (~fix_mask)
            x = torch.where(remaining_masks, sampled_x0, x)
            
            all_samples.append(x.detach().cpu())
            remaining -= curr_batch
    
    return torch.cat(all_samples, dim=0)


def compute_perplexity(
    texts: List[str],
    eval_model: AutoModelForCausalLM,
    eval_tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict:
    """Compute generative perplexity using the eval model.
    
    Args:
        texts: List of generated text samples
        eval_model: Pretrained causal LM for evaluation
        eval_tokenizer: Tokenizer for eval model
        batch_size: Batch size for evaluation
        max_length: Max sequence length
        device: Device to run on
        
    Returns:
        Dict with perplexity metrics
    """
    total_nll = 0.0
    total_tokens = 0
    all_nlls = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize with eval model's tokenizer
            batch = eval_tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = eval_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1]  # Remove last position
            labels = input_ids[:, 1:]  # Shift targets
            loss_mask = attention_mask[:, :-1]
            
            # Compute per-token NLL
            nll = F.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(0, 1),
                reduction="none",
            ).view_as(labels)
            
            # Only count valid (non-padded) tokens
            valid = loss_mask.bool()
            valid_nlls = nll[valid].detach().cpu().numpy().tolist()
            all_nlls.extend(valid_nlls)
            
            total_nll += float((nll * loss_mask).sum().item())
            total_tokens += int(valid.sum().item())
    
    if total_tokens == 0:
        return {
            "avg_nll": float("nan"),
            "ppl": float("nan"),
            "median_nll": float("nan"),
            "tokens_evaluated": 0,
        }
    
    avg_nll = total_nll / total_tokens
    ppl = float(np.exp(avg_nll))
    median_nll = float(np.median(all_nlls)) if all_nlls else float("nan")
    
    return {
        "avg_nll": float(avg_nll),
        "ppl": ppl,
        "median_nll": median_nll,
        "tokens_evaluated": total_tokens,
    }


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)
    
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Num steps: {args.num_steps}")
    print(f"  Prompt mode: {args.prompt_mode}")
    if args.prompt_mode != "none":
        print(f"  Prompt text: {args.prompt_text}")
        if args.prompt_mode == "infill":
            print(f"  Suffix text: {args.suffix_text}")
        print(f"  Generated length: {args.generated_length}")
    
    # Load MDLM model
    model, tokenizer, config = load_mdlm_model(args.checkpoint_path, device)
    
    # Load evaluation model
    print(f"\nLoading evaluation model: {args.eval_model}")
    eval_model = AutoModelForCausalLM.from_pretrained(
        args.eval_model,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    eval_tokenizer = AutoTokenizer.from_pretrained(args.eval_model)
    if eval_tokenizer.pad_token_id is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
    eval_model.eval()
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with {args.num_steps} steps...")
    
    if args.prompt_mode == "none":
        samples = generate_samples_unconditional(
            model,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
        )
    else:
        if not args.prompt_text:
            raise ValueError("prompt_text is required for prefix/infill modes")
        samples = generate_samples_with_prompt(
            model,
            tokenizer,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            prompt_text=args.prompt_text,
            suffix_text=args.suffix_text if args.prompt_mode == "infill" else None,
            generated_length=args.generated_length,
            batch_size=args.batch_size,
        )
    
    # Decode to text
    texts = tokenizer.batch_decode(samples, skip_special_tokens=True)
    
    # Print sample preview
    print(f"\nSample preview (first 300 chars):")
    print(f"  {texts[0][:300]}...")
    
    # Compute perplexity
    print(f"\nComputing perplexity with {args.eval_model}...")
    metrics = compute_perplexity(
        texts=texts,
        eval_model=eval_model,
        eval_tokenizer=eval_tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    
    print(f"\nResults:")
    print(f"  PPL: {metrics['ppl']:.2f}")
    print(f"  Avg NLL: {metrics['avg_nll']:.4f}")
    print(f"  Median NLL: {metrics['median_nll']:.4f}")
    print(f"  Tokens evaluated: {metrics['tokens_evaluated']}")
    
    # Compile output
    output = {
        "checkpoint_path": args.checkpoint_path,
        "eval_model": args.eval_model,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "max_length": args.max_length,
        "prompt_mode": args.prompt_mode,
        "prompt_text": args.prompt_text if args.prompt_mode != "none" else None,
        "suffix_text": args.suffix_text if args.prompt_mode == "infill" else None,
        "generated_length": args.generated_length if args.prompt_mode != "none" else None,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    
    if args.save_samples:
        output["samples"] = texts
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

