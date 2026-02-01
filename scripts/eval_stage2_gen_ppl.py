#!/usr/bin/env python3
"""Evaluate Stage 2 decoder checkpoints with Generative Perplexity.

This script:
1. Loads each best.ckpt from the Stage 2 decoder runs
2. Generates sequences using the LatentJEPASampler
3. Computes generative perplexity using GPT2-XL
4. Outputs aggregated results as JSON

Usage:
    # Using config file (recommended):
    python scripts/eval_stage2_gen_ppl.py \
        --checkpoint_dir /path/to/checkpoints \
        --sampling_config configs/sampling/latent_jepa.yaml \
        --num_samples 10 \
        --output_path gen_ppl_results.json

    # CLI args override config values:
    python scripts/eval_stage2_gen_ppl.py \
        --checkpoint_dir /path/to/checkpoints \
        --sampling_config configs/sampling/latent_jepa.yaml \
        --temperature 0.8 \
        --top_p 0.95
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra.utils
from discrete_diffusion.data import get_tokenizer
from discrete_diffusion.sampling.latent_jepa import LatentJEPASampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 2 checkpoints with Generative Perplexity"
    )
    # Required arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing Stage 2 checkpoint subdirectories",
    )
    # Config file (sampling parameters loaded from here, CLI args override)
    parser.add_argument(
        "--sampling_config",
        type=str,
        default=None,
        help="Path to sampling config YAML (e.g., configs/sampling/latent_jepa.yaml). CLI args override config values.",
    )
    # Evaluation arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate per checkpoint",
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
        help="Batch size for perplexity evaluation",
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
        "--checkpoint_pattern",
        type=str,
        default="*/checkpoints/best.ckpt",
        help="Glob pattern to find checkpoints within checkpoint_dir",
    )
    # Sampling options (override config if provided)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of denoising steps for generation (overrides config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling probability (overrides config)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling, 0 to disable (overrides config)",
    )
    parser.add_argument(
        "--commit_schedule",
        type=str,
        default=None,
        choices=["hazard", "uniform"],
        help="Commit schedule for selecting how many masked tokens to fill per step (overrides config)",
    )
    parser.add_argument(
        "--commit_fraction",
        type=float,
        default=None,
        help="Override fraction of remaining masks to commit per step (overrides config)",
    )
    parser.add_argument(
        "--min_tokens_to_keep",
        type=int,
        default=None,
        help="Minimum tokens to keep when applying top-p sampling (overrides config)",
    )
    parser.add_argument(
        "--ban_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mask special tokens (pad/mask) from sampling logits (overrides config)",
    )
    # Prompt-based generation options (override config if provided)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=None,
        choices=["none", "prefix", "infill"],
        help="Prompt mode: 'none' (unconditional), 'prefix' (left-to-right completion), 'infill' (fill middle) (overrides config)",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="Prompt text for prefix mode, or prefix text for infill mode (overrides config)",
    )
    parser.add_argument(
        "--suffix_text",
        type=str,
        default=None,
        help="Suffix text for infill mode (ignored in other modes) (overrides config)",
    )
    parser.add_argument(
        "--generated_length",
        type=int,
        default=None,
        help="Number of tokens to generate (for prefix mode) or infill (for infill mode) (overrides config)",
    )
    return parser.parse_args()


def load_sampling_config(config_path: Optional[str], args) -> dict:
    """Load sampling config from YAML and merge with CLI args.
    
    Args:
        config_path: Path to YAML config file (or None)
        args: Parsed CLI arguments
        
    Returns:
        Dict with final sampling parameters (CLI args override config)
    """
    # Default values
    defaults = {
        "steps": 64,
        "commit_schedule": "hazard",
        "commit_fraction": None,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "min_tokens_to_keep": 1,
        "ban_special_tokens": False,
        "inject_bos": True,
        "prompt_mode": "none",
        "prompt_text": None,
        "suffix_text": None,
        "generated_length": 256,
    }
    
    # Load from config file if provided
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Sampling config not found: {config_path}")
        
        print(f"Loading sampling config from: {config_path}")
        cfg = OmegaConf.load(config_file)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Update defaults with config values
        for key in defaults:
            if key in cfg_dict and cfg_dict[key] is not None:
                defaults[key] = cfg_dict[key]
    
    # CLI args override config values (only if explicitly provided)
    cli_overrides = {
        "steps": args.num_steps,
        "commit_schedule": args.commit_schedule,
        "commit_fraction": args.commit_fraction,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_tokens_to_keep": args.min_tokens_to_keep,
        "ban_special_tokens": args.ban_special_tokens,
        "prompt_mode": args.prompt_mode,
        "prompt_text": args.prompt_text,
        "suffix_text": args.suffix_text,
        "generated_length": args.generated_length,
    }
    
    for key, value in cli_overrides.items():
        if value is not None:
            defaults[key] = value
    
    return defaults


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


class SamplerModelWrapper:
    """Wrapper that combines trainer attributes with backbone methods for the sampler."""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.backbone = trainer.backbone
        # Expose trainer attributes needed by sampler
        self.mask_id = trainer.mask_id
        self.tokenizer = trainer.tokenizer
        self.num_tokens = trainer.num_tokens
        self.max_seq_len = getattr(trainer.backbone, 'max_seq_len', trainer.num_tokens)
        
        # Expose latent normalization for consistent decoder input
        # If trainer has normalize_targets=True, expose the target_norm layer
        self.latent_norm = None
        if getattr(trainer, 'normalize_targets', False) and hasattr(trainer, 'target_norm'):
            self.latent_norm = trainer.target_norm
    
    def parameters(self):
        return self.backbone.parameters()
    
    def eval(self):
        self.backbone.eval()
        return self
    
    def train(self):
        self.backbone.train()
        return self
    
    # Forward backbone methods
    def encode_student(self, *args, **kwargs):
        return self.backbone.encode_student(*args, **kwargs)
    
    def predict_latent(self, *args, **kwargs):
        return self.backbone.predict_latent(*args, **kwargs)
    
    def readout_tokens(self, *args, **kwargs):
        return self.backbone.readout_tokens(*args, **kwargs)
    
    def normalize_latent(self, z):
        """Apply latent normalization if enabled during training."""
        if self.latent_norm is not None:
            return self.latent_norm(z)
        return z


def generate_with_prompt(
    wrapped_model,
    num_samples: int,
    num_steps: int,
    prompt_ids: Optional[torch.Tensor] = None,
    suffix_ids: Optional[torch.Tensor] = None,
    total_length: int = 256,
    eps: float = 1e-5,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    commit_schedule: str = "uniform",
    commit_fraction: Optional[float] = None,
    min_tokens_to_keep: int = 1,
    ban_special_tokens: bool = True,
) -> torch.Tensor:
    """Generate samples with optional prompt (prefix) and suffix (for infilling).
    
    Args:
        wrapped_model: SamplerModelWrapper instance
        num_samples: Number of samples to generate
        num_steps: Number of denoising steps
        prompt_ids: Token IDs for the prefix prompt [prompt_len]
        suffix_ids: Token IDs for the suffix (infill mode) [suffix_len]
        total_length: Total sequence length to generate
        eps: Small epsilon for timestep bounds
        temperature: Sampling temperature
        top_p: Nucleus sampling p
        top_k: Top-k sampling k
        commit_schedule: Strategy for deciding how many masked tokens to commit per step
        commit_fraction: Override fraction of remaining masks committed per step
        min_tokens_to_keep: Minimum tokens to keep for top-p filtering
        ban_special_tokens: Whether to mask special tokens (pad/mask) from sampling
        
    Returns:
        Tensor of token IDs [num_samples, total_length]
    """
    from discrete_diffusion.sampling.position_scorer import ConfidencePositionScorer
    from discrete_diffusion.sampling.token_selection import GreedySelection, NucleusSelection, TopKSelection, TemperatureSelection
    
    wrapped_model.eval()
    device = next(wrapped_model.parameters()).device
    mask_id = wrapped_model.mask_id
    tokenizer = wrapped_model.tokenizer
    
    position_scorer = ConfidencePositionScorer()
    
    if top_p < 1.0:
        token_selector = NucleusSelection(
            p=top_p,
            temperature=temperature,
            min_tokens_to_keep=min_tokens_to_keep,
        )
    elif top_k > 0:
        token_selector = TopKSelection(k=top_k, temperature=temperature)
    elif temperature != 1.0:
        token_selector = TemperatureSelection(temperature=temperature)
    else:
        token_selector = GreedySelection()
    
    # Initialize sequence with all masks
    x = torch.full((num_samples, total_length), mask_id, dtype=torch.long, device=device)
    
    # Set up fixed positions mask (positions that should NOT be denoised)
    fix_mask = torch.zeros((num_samples, total_length), dtype=torch.bool, device=device)
    
    # Handle prompt (prefix)
    prompt_len = 0
    if prompt_ids is not None and len(prompt_ids) > 0:
        prompt_len = min(len(prompt_ids), total_length - 1)
        x[:, :prompt_len] = prompt_ids[:prompt_len].unsqueeze(0).expand(num_samples, -1)
        fix_mask[:, :prompt_len] = True
    
    # Handle suffix (for infill mode)
    suffix_len = 0
    if suffix_ids is not None and len(suffix_ids) > 0:
        suffix_len = min(len(suffix_ids), total_length - prompt_len - 1)
        suffix_start = total_length - suffix_len
        x[:, suffix_start:] = suffix_ids[-suffix_len:].unsqueeze(0).expand(num_samples, -1)
        fix_mask[:, suffix_start:] = True
    
    banned_token_ids = None
    if ban_special_tokens:
        banned_token_ids = {int(mask_id)}
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            banned_token_ids.add(int(pad_id))
    if banned_token_ids:
        banned_token_ids = sorted(banned_token_ids)

    # Timestep schedule
    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    
    for i in range(num_steps):
        mask_index = (x == mask_id) & (~fix_mask)
        if not mask_index.any():
            break
        
        t, s = timesteps[i], timesteps[i + 1]
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.float32)
        
        # Latent forward pass
        z_t = wrapped_model.encode_student(x, t_batch)
        z_hat_0 = wrapped_model.predict_latent(z_t, t_batch)
        
        # Apply latent normalization before readout (consistent with training)
        z_hat_0 = wrapped_model.normalize_latent(z_hat_0)
        
        logits = wrapped_model.readout_tokens(z_hat_0)
        
        # Score positions
        if banned_token_ids:
            logits[..., banned_token_ids] = -float("inf")
        scores = position_scorer(logits, device)
        scores = scores.masked_fill(~mask_index, -float("inf"))
        
        # Determine how many positions to transfer per sequence
        masked_counts = mask_index.sum(dim=-1)
        if commit_fraction is not None:
            p_transfer = commit_fraction
            num_to_transfer = torch.ceil(masked_counts.float() * p_transfer).to(torch.long)
        elif commit_schedule == "uniform":
            remaining_steps = max(1, num_steps - i)
            num_to_transfer = torch.ceil(masked_counts.float() / remaining_steps).to(torch.long)
        elif commit_schedule == "hazard":
            p_transfer = 1 - s / t if i < num_steps - 1 else 1.0
            num_to_transfer = torch.ceil(masked_counts.float() * p_transfer).to(torch.long)
        else:
            raise ValueError(f"Unknown commit_schedule: {commit_schedule}")
        num_to_transfer = torch.minimum(num_to_transfer, masked_counts)
        
        batch_size, seq_len = x.shape
        
        # Rank positions by score
        order = torch.argsort(scores, dim=-1, descending=True)
        
        # Build boolean mask selecting top-k per row
        rank_threshold = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1) < num_to_transfer.unsqueeze(1)
        
        select_mask = torch.zeros_like(mask_index, dtype=torch.bool)
        select_mask.scatter_(1, order, rank_threshold)
        select_mask = select_mask & mask_index
        
        # Select and commit tokens
        if select_mask.any():
            selected_logits = logits[select_mask]
            selected_tokens = token_selector(selected_logits)
            x[select_mask] = selected_tokens
    
    return x


def generate_samples(
    model,
    config,
    num_samples: int,
    num_steps: int,
    batch_size: int = 8,
    prompt_mode: str = "none",
    prompt_text: str = "",
    suffix_text: str = "",
    generated_length: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    commit_schedule: str = "uniform",
    commit_fraction: Optional[float] = None,
    min_tokens_to_keep: int = 1,
    ban_special_tokens: bool = True,
) -> torch.Tensor:
    """Generate samples from the model.
    
    Args:
        model: The loaded LatentJEPATrainer model
        config: Model config (for sampler initialization)
        num_samples: Total number of samples to generate
        num_steps: Number of denoising steps
        batch_size: Batch size for generation
        prompt_mode: 'none', 'prefix', or 'infill'
        prompt_text: Prompt text for prefix/infill modes
        suffix_text: Suffix text for infill mode
        generated_length: Length of generated/infilled tokens
        temperature: Sampling temperature
        top_p: Nucleus sampling p
        top_k: Top-k sampling k
        commit_schedule: Strategy for deciding how many masked tokens to commit per step
        commit_fraction: Override fraction of remaining masks committed per step
        min_tokens_to_keep: Minimum tokens to keep for top-p filtering
        ban_special_tokens: Whether to mask special tokens (pad/mask) from sampling
        
    Returns:
        Tensor of token IDs [num_samples, seq_length]
    """
    # Wrap model to provide the interface the sampler expects
    wrapped_model = SamplerModelWrapper(model)
    device = next(wrapped_model.parameters()).device
    tokenizer = wrapped_model.tokenizer
    
    from discrete_diffusion.sampling.token_selection import GreedySelection, NucleusSelection, TopKSelection, TemperatureSelection

    if top_p < 1.0:
        token_selector = NucleusSelection(
            p=top_p,
            temperature=temperature,
            min_tokens_to_keep=min_tokens_to_keep,
        )
    elif top_k > 0:
        token_selector = TopKSelection(k=top_k, temperature=temperature)
    elif temperature != 1.0:
        token_selector = TemperatureSelection(temperature=temperature)
    else:
        token_selector = GreedySelection()

    # Tokenize prompts if needed
    prompt_ids = None
    suffix_ids = None
    
    if prompt_mode == "prefix" and prompt_text:
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt_text, add_special_tokens=False),
            dtype=torch.long,
            device=device
        )
        total_length = len(prompt_ids) + generated_length
        
    elif prompt_mode == "infill" and prompt_text:
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt_text, add_special_tokens=False),
            dtype=torch.long,
            device=device
        )
        if suffix_text:
            suffix_ids = torch.tensor(
                tokenizer.encode(suffix_text, add_special_tokens=False),
                dtype=torch.long,
                device=device
            )
        total_length = len(prompt_ids) + generated_length + (len(suffix_ids) if suffix_ids is not None else 0)
        
    else:  # none - unconditional generation
        total_length = model.num_tokens
    
    # Cap total length to model's max
    total_length = min(total_length, model.num_tokens)
    
    all_samples = []
    remaining = num_samples
    
    with torch.no_grad():
        while remaining > 0:
            curr_batch = min(batch_size, remaining)
            
            if prompt_mode == "none":
                # Use original sampler for unconditional generation
                # Pass latent_norm for consistent normalization with training
                sampler = LatentJEPASampler(
                    config,
                    token_selector=token_selector,
                    commit_schedule=commit_schedule,
                    commit_fraction=commit_fraction,
                    min_tokens_to_keep=min_tokens_to_keep,
                    ban_special_tokens=ban_special_tokens,
                    latent_norm=wrapped_model.latent_norm,
                )
                inject_bos = getattr(config.sampling, "inject_bos", True)
                samples = sampler.generate(
                    model=wrapped_model,
                    num_samples=curr_batch,
                    num_steps=num_steps,
                    eps=1e-5,
                    inject_bos=inject_bos,
                )
            else:
                # Use prompted generation
                samples = generate_with_prompt(
                    wrapped_model=wrapped_model,
                    num_samples=curr_batch,
                    num_steps=num_steps,
                    prompt_ids=prompt_ids,
                    suffix_ids=suffix_ids,
                    total_length=total_length,
                    eps=1e-5,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    commit_schedule=commit_schedule,
                    commit_fraction=commit_fraction,
                    min_tokens_to_keep=min_tokens_to_keep,
                    ban_special_tokens=ban_special_tokens,
                )
            
            all_samples.append(samples.detach().cpu())
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
    
    # Load sampling config
    sampling_cfg = load_sampling_config(args.sampling_config, args)
    
    print("\nSampling configuration:")
    print(f"  steps: {sampling_cfg['steps']}")
    print(f"  commit_schedule: {sampling_cfg['commit_schedule']}")
    print(f"  temperature: {sampling_cfg['temperature']}")
    print(f"  top_p: {sampling_cfg['top_p']}")
    print(f"  top_k: {sampling_cfg['top_k']}")
    print(f"  prompt_mode: {sampling_cfg['prompt_mode']}")
    if sampling_cfg['prompt_mode'] != 'none':
        print(f"  prompt_text: {sampling_cfg['prompt_text']}")
        if sampling_cfg['prompt_mode'] == 'infill':
            print(f"  suffix_text: {sampling_cfg['suffix_text']}")
        print(f"  generated_length: {sampling_cfg['generated_length']}")
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir, args.checkpoint_pattern)
    if not checkpoints:
        print(f"No checkpoints found at {args.checkpoint_dir}/{args.checkpoint_pattern}")
        sys.exit(1)
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.parent.parent.name}")
    
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
            
            # Generate samples
            mode_str = f" ({sampling_cfg['prompt_mode']} mode)" if sampling_cfg['prompt_mode'] != "none" else ""
            print(f"Generating {args.num_samples} samples with {sampling_cfg['steps']} steps{mode_str}...")
            if sampling_cfg['prompt_mode'] == "prefix":
                print(f"  Prompt: '{sampling_cfg['prompt_text']}'")
            elif sampling_cfg['prompt_mode'] == "infill":
                print(f"  Prefix: '{sampling_cfg['prompt_text']}'")
                print(f"  Suffix: '{sampling_cfg['suffix_text']}'")
            
            samples = generate_samples(
                model,
                config,
                num_samples=args.num_samples,
                num_steps=sampling_cfg['steps'],
                batch_size=min(args.num_samples, 8),
                prompt_mode=sampling_cfg['prompt_mode'],
                prompt_text=sampling_cfg['prompt_text'] or "",
                suffix_text=sampling_cfg['suffix_text'] or "",
                generated_length=sampling_cfg['generated_length'],
                temperature=sampling_cfg['temperature'],
                top_p=sampling_cfg['top_p'],
                top_k=sampling_cfg['top_k'],
                commit_schedule=sampling_cfg['commit_schedule'],
                commit_fraction=sampling_cfg['commit_fraction'],
                min_tokens_to_keep=sampling_cfg['min_tokens_to_keep'],
                ban_special_tokens=sampling_cfg['ban_special_tokens'],
            )
            
            # Decode to text using the model's tokenizer
            texts = model_tokenizer.batch_decode(samples, skip_special_tokens=True)
            
            # Print sample preview
            print(f"\nSample preview (first 200 chars):")
            print(f"  {texts[0][:200]}...")
            
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
            
            result = {
                "checkpoint": run_name,
                "checkpoint_path": str(ckpt_path),
                **metrics,
            }
            
            if args.save_samples:
                result["samples"] = texts
            
            results.append(result)
            
            print(f"\nResults for {run_name}:")
            print(f"  PPL: {metrics['ppl']:.2f}")
            print(f"  Avg NLL: {metrics['avg_nll']:.4f}")
            print(f"  Median NLL: {metrics['median_nll']:.4f}")
            print(f"  Tokens evaluated: {metrics['tokens_evaluated']}")
            
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
        "eval_model": args.eval_model,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
        "sampling_config": args.sampling_config,
        # Sampling parameters (from config + CLI overrides)
        "num_steps": sampling_cfg['steps'],
        "temperature": sampling_cfg['temperature'],
        "top_p": sampling_cfg['top_p'],
        "top_k": sampling_cfg['top_k'],
        "commit_schedule": sampling_cfg['commit_schedule'],
        "commit_fraction": sampling_cfg['commit_fraction'],
        "min_tokens_to_keep": sampling_cfg['min_tokens_to_keep'],
        "ban_special_tokens": sampling_cfg['ban_special_tokens'],
        "prompt_mode": sampling_cfg['prompt_mode'],
        "prompt_text": sampling_cfg['prompt_text'] if sampling_cfg['prompt_mode'] != "none" else None,
        "suffix_text": sampling_cfg['suffix_text'] if sampling_cfg['prompt_mode'] == "infill" else None,
        "generated_length": sampling_cfg['generated_length'] if sampling_cfg['prompt_mode'] != "none" else None,
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
    print("-" * 80)
    print(f"{'Checkpoint':<50} {'PPL':>10} {'Avg NLL':>10}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['checkpoint']:<50} {'ERROR':>10}")
        else:
            print(f"{r['checkpoint']:<50} {r['ppl']:>10.2f} {r['avg_nll']:>10.4f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
