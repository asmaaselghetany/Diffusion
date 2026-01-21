#!/usr/bin/env python
"""MTEB evaluation script for JEPA embedding models."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "mteb"))

import mteb
from mteb.models import ModelMeta
from mteb.types import PromptType
from torch.utils.data import DataLoader

from src.discrete_diffusion.data import get_tokenizer


class JEPAEncoder:
    """MTEB-compatible wrapper for JEPA embedding models."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        encoder: str = "teacher",
        pooling: str = "mean",
        device: str = "cuda",
        max_length: int | None = None,
        l2_normalize: bool = False,
        eval_timestep: float = 1e-3,  # Must match training sampling_eps!
        use_predictor: bool = False,  # Use predictor output instead of raw encoder
        apply_target_norm: bool = False,  # Apply LayerNorm to encoder outputs (like training)
    ):
        self.device = device
        self.encoder_type = encoder
        self.pooling = pooling
        self.l2_normalize = l2_normalize
        self.eval_timestep = eval_timestep  # Timestep used for encoding clean text
        self.use_predictor = use_predictor
        self.apply_target_norm = apply_target_norm

        # Load checkpoint to extract config
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract config from checkpoint hyper_parameters
        if "hyper_parameters" in ckpt and "config" in ckpt["hyper_parameters"]:
            self.config = ckpt["hyper_parameters"]["config"]
        elif config_path:
            self.config = OmegaConf.load(config_path)
        else:
            raise ValueError("Config not found in checkpoint and --config not provided")
        
        # Default max length to the training config if not provided
        if max_length is None:
            max_length = getattr(self.config.model, "length", 512)
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = get_tokenizer(self.config)
        
        # Get algorithm class and load checkpoint
        algo_cls = hydra.utils.get_class(self.config.algo._target_)
        self.model = algo_cls.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
            tokenizer=self.tokenizer,
            strict=False,
            map_location=device,
        )
        self.model.to(device)
        self.model.eval()
        
        # Get backbone (LatentJEPA)
        self.backbone = self.model.backbone
        self.latent_dim = self.backbone.latent_dim
        
        # Try to get target_norm from the trainer (used during training to normalize targets)
        # This is critical: during training, targets are normalized via LayerNorm!
        self.target_norm = getattr(self.model, 'target_norm', None)
        if self.target_norm is not None:
            self.target_norm = self.target_norm.to(device)
            print(f"  Found trained target_norm in checkpoint")
        elif apply_target_norm:
            # Create a new LayerNorm (weights will be default, not trained)
            self.target_norm = torch.nn.LayerNorm(self.latent_dim).to(device)
            print(f"  WARNING: Using fresh LayerNorm (not trained weights)")
        
        # Get pad token id
        self.pad_id = getattr(self.tokenizer, 'pad_token_id', None)
        if self.pad_id is None:
            self.pad_id = getattr(self.tokenizer, 'eos_token_id', 0)
        
        # MTEB model metadata (required for MTEB to recognize as custom model)
        n_params = sum(p.numel() for p in self.backbone.parameters())
        enc_name = "predictor" if use_predictor else encoder
        norm_suffix = "-l2norm" if l2_normalize else ("-targetnorm" if apply_target_norm else "-raw")
        self.mteb_model_meta = ModelMeta(
            name=f"local/jepa-{enc_name}-{pooling}{norm_suffix}",
            revision="1.0.0",
            languages=["eng-Latn"],
            open_weights=True,
            release_date="2025-01-01",
            n_parameters=n_params,
            memory_usage_mb=int(n_params * 4 / 1024 / 1024),  # Approximate
            embed_dim=self.latent_dim,
            license="apache-2.0",
            max_tokens=max_length,
            similarity_fn_name="cosine",
            framework=["PyTorch"],
            use_instructions=False,
            loader=lambda *args, **kwargs: self,  # Return self as loader
            public_training_code=None,
            public_training_data=None,
            training_datasets=None,
        )

    def _pool(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool sequence embeddings to single vector.
        
        Args:
            embeddings: (B, L, D) token embeddings
            attention_mask: (B, L) mask where 1 = valid token
            
        Returns:
            (B, D) pooled embeddings
        """
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "cls":
            return embeddings[:, 0]
        elif self.pooling == "last":
            # Get last valid token for each sequence
            seq_lens = attention_mask.sum(dim=1).long() - 1
            batch_idx = torch.arange(embeddings.size(0), device=embeddings.device)
            return embeddings[batch_idx, seq_lens]
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = embeddings.masked_fill(mask == 0, float('-inf'))
            return embeddings.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize texts and return input_ids and attention_mask."""
        if hasattr(self.tokenizer, 'encode_batch'):
            # Custom tokenizer (e.g., Text8)
            input_ids = []
            for text in texts:
                ids = self.tokenizer.encode(text)
                if len(ids) > self.max_length:
                    ids = ids[:self.max_length]
                input_ids.append(ids)
            # Pad to max length in batch
            max_len = max(len(ids) for ids in input_ids)
            attention_mask = []
            padded_ids = []
            for ids in input_ids:
                pad_len = max_len - len(ids)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
                padded_ids.append(ids + [self.pad_id] * pad_len)
            return (
                torch.tensor(padded_ids, device=self.device),
                torch.tensor(attention_mask, device=self.device),
            )
        else:
            # HuggingFace tokenizer
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return (
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
            )

    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embeddings."""
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return embeddings1 @ embeddings2.T

    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity."""
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return (embeddings1 * embeddings2).sum(axis=1)

    @torch.no_grad()
    def encode(
        self,
        inputs: DataLoader,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []
        
        for batch in inputs:
            texts = batch["text"]
            input_ids, attention_mask = self._tokenize(texts)
            
            # Create timestep tensor for time-conditioned encoding
            # CRITICAL: Must use eval_timestep (default 1e-3) instead of t=0!
            # The model was trained with t ∈ [sampling_eps, 1-sampling_eps], so t=0 is OOD.
            batch_size = input_ids.shape[0]
            t = torch.full((batch_size,), self.eval_timestep, device=self.device, dtype=torch.float32)
            
            # Get latent embeddings with proper timestep conditioning
            if self.use_predictor:
                # Use predictor output: predictor was trained to match target_norm(teacher(clean_input))
                # So predictor output is already in the "normalized" representation space
                if self.encoder_type == "teacher":
                    z_enc = self.backbone.encode_teacher(input_ids, t, attention_mask=attention_mask)
                else:
                    z_enc = self.backbone.encode_student(input_ids, t, attention_mask=attention_mask)
                # Pass through predictor to get refined representations
                latents = self.backbone.predict_latent(z_enc, t)
            else:
                # Use raw encoder output
                if self.encoder_type == "teacher":
                    latents = self.backbone.encode_teacher(input_ids, t, attention_mask=attention_mask)
                else:
                    latents = self.backbone.encode_student(input_ids, t, attention_mask=attention_mask)
                
                # Apply target normalization if requested (matches training target processing)
                # During training: target = target_norm(teacher_encoder(clean_input))
                if self.apply_target_norm and self.target_norm is not None:
                    latents = self.target_norm(latents)
            
            # Pool to single vector per sequence
            pooled = self._pool(latents, attention_mask)
            
            # Apply L2 normalization if enabled
            if self.l2_normalize:
                pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-8)
            
            all_embeddings.append(pooled.cpu().float().numpy())
        
        return np.concatenate(all_embeddings, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JEPA model on MTEB benchmarks")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to JEPA checkpoint (.ckpt)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to Hydra config (.yaml). If not provided, uses config from checkpoint.")
    
    # Encoder options
    parser.add_argument("--encoder", type=str, default="teacher",
                        choices=["teacher", "student"],
                        help="Which encoder to use (default: teacher)")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls", "last", "max"],
                        help="Pooling strategy (default: mean)")
    parser.add_argument("--l2-normalize", action="store_true",
                        help="Apply L2 normalization to embeddings (recommended for clustering/classification)")
    parser.add_argument("--eval-timestep", type=float, default=1e-3,
                        help="Timestep for encoding (default: 1e-3 = sampling_eps). "
                             "CRITICAL: Must match training sampling_eps to avoid OOD embeddings!")
    parser.add_argument("--use-predictor", action="store_true",
                        help="Use predictor output instead of raw encoder. "
                             "The predictor was trained to match normalized targets, so this may give better representations.")
    parser.add_argument("--apply-target-norm", action="store_true",
                        help="Apply LayerNorm to encoder outputs (matches training target processing). "
                             "Use with trained target_norm weights from checkpoint if available.")
    
    # Task selection
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific task names to evaluate (e.g., STS12 STSBenchmark)")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="MTEB benchmark name (e.g., 'MTEB(eng, v2)')")
    parser.add_argument("--task-types", type=str, nargs="+", default=None,
                        choices=["STS", "Classification", "Clustering", "Retrieval",
                                 "Reranking", "PairClassification", "BitextMining"],
                        help="Filter tasks by type")
    parser.add_argument("--languages", type=str, nargs="+", default=["eng"],
                        help="Filter tasks by language (default: eng)")
    
    # Runtime options
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for encoding (default: 32)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Maximum sequence length (default: model config length)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="mteb_results",
                        help="Directory to save results (default: mteb_results)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build output name suffix
    if args.l2_normalize:
        norm_str = "l2norm"
    elif args.apply_target_norm:
        norm_str = "targetnorm"
    else:
        norm_str = "raw"
    
    enc_str = "predictor" if args.use_predictor else args.encoder
    
    print(f"Loading JEPA model from {args.checkpoint}")
    print(f"  Config: {args.config}")
    print(f"  Encoder: {args.encoder}, Pooling: {args.pooling}")
    print(f"  Use predictor: {args.use_predictor}, Apply target norm: {args.apply_target_norm}")
    print(f"  L2 Normalize: {args.l2_normalize}")
    print(f"  Eval timestep: {args.eval_timestep} (should match training sampling_eps)")
    
    # Initialize model wrapper
    model = JEPAEncoder(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        encoder=args.encoder,
        pooling=args.pooling,
        device=args.device,
        max_length=args.max_length,
        l2_normalize=args.l2_normalize,
        eval_timestep=args.eval_timestep,
        use_predictor=args.use_predictor,
        apply_target_norm=args.apply_target_norm,
    )
    print(f"  Latent dim: {model.latent_dim}")
    
    # Get tasks
    if args.benchmark:
        print(f"Loading benchmark: {args.benchmark}")
        benchmark = mteb.get_benchmark(args.benchmark)
        tasks = benchmark.tasks
    elif args.tasks:
        print(f"Loading tasks: {args.tasks}")
        tasks = mteb.get_tasks(tasks=args.tasks)
    else:
        print(f"Loading tasks by type={args.task_types}, languages={args.languages}")
        tasks = mteb.get_tasks(
            task_types=args.task_types,
            languages=args.languages,
        )
    
    print(f"Evaluating on {len(list(tasks))} tasks")
    
    # Run evaluation
    results = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs={"batch_size": args.batch_size},
        raise_error=False,
    )
    
    # Save and print results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build summary
    summary = {
        "checkpoint": args.checkpoint,
        "encoder": args.encoder,
        "pooling": args.pooling,
        "l2_normalize": args.l2_normalize,
        "model_name": results.model_name,
        "scores": {},
    }
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for task_result in results.task_results:
        task_name = task_result.task_name
        main_score = task_result.get_score()
        summary["scores"][task_name] = {
            "main_score": main_score,
            "all_scores": task_result.scores,
        }
        print(f"{task_name}: {main_score:.4f}")
    
    # Save full results
    output_file = output_dir / f"results_{enc_str}_{args.pooling}_{norm_str}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    
    # Print any errors
    if results.exceptions:
        print("\nErrors:")
        for exc in results.exceptions:
            print(f"  {exc.task_name}: {exc.exception}")


if __name__ == "__main__":
    main()
