"""
Qwen3 Embedding provider for continuous embedding diffusion.

This provider wraps a pretrained Qwen3-Embedding model to extract
dense embeddings that can be used as the basis for continuous diffusion.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Qwen3EmbeddingProvider(nn.Module):
    """
    Qwen3-Embedding provider for extracting dense text embeddings.
    
    Loads a pretrained Qwen3-Embedding model and extracts dense embeddings
    that can be used as input to continuous Gaussian diffusion.
    
    Features:
    - Frozen by default (recommended for stability)
    - Multiple pooling strategies (token-level, mean, last, cls)
    - Support for instruction-aware embedding (Qwen-specific)
    - bf16/fp16 support for memory efficiency
    - Optional output projection to different dimension
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        pooling: str = "token",
        trainable: bool = False,
        dtype: str = "bfloat16",
        output_dim: Optional[int] = None,
        use_instruction: bool = False,
        instruction_template: str = "Instruct: {task}\nQuery: {text}",
        default_task: str = "Represent this text for retrieval",
    ):
        """
        Args:
            pretrained_model_name: HuggingFace model name or path
            pooling: Pooling strategy - 'token', 'mean', 'last', 'cls'
            trainable: Whether to train the encoder (default: frozen)
            dtype: Model dtype - 'float32', 'float16', 'bfloat16'
            output_dim: Optional output dimension (projects from hidden_size)
            use_instruction: Whether to prepend instruction to input
            instruction_template: Template for instruction-aware embedding
            default_task: Default task description for instruction mode
        """
        super().__init__()
        
        self.pretrained_model_name = pretrained_model_name
        self.pooling = pooling
        self.trainable = trainable
        self.use_instruction = use_instruction
        self.instruction_template = instruction_template
        self.default_task = default_task
        
        # Validate pooling
        valid_poolings = ["token", "mean", "last", "cls"]
        if pooling not in valid_poolings:
            raise ValueError(f"Invalid pooling: {pooling}. Must be one of: {valid_poolings}")
        
        # Set dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Invalid dtype: {dtype}. Must be one of: {list(dtype_map.keys())}")
        self.model_dtype = dtype_map[dtype]
        
        # Load pretrained model
        print(f"Loading Qwen3-Embedding provider from {pretrained_model_name}...")
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name,
            torch_dtype=self.model_dtype,
            trust_remote_code=True,
        )
        self.hidden_size = self.encoder.config.hidden_size
        
        # Optional output projection
        self.output_dim = output_dim or self.hidden_size
        if output_dim is not None and output_dim != self.hidden_size:
            self.output_proj = nn.Linear(self.hidden_size, output_dim)
        else:
            self.output_proj = None
        
        # Freeze encoder if not trainable
        if not trainable:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        print(f"Qwen3-Embedding loaded: hidden_size={self.hidden_size}, "
              f"output_dim={self.output_dim}, trainable={trainable}")
    
    def _pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply pooling strategy to hidden states.
        
        Args:
            hidden_states: [B, L, D] encoder hidden states
            attention_mask: [B, L] attention mask (1 for valid tokens, 0 for padding)
            
        Returns:
            Pooled embeddings:
            - If pooling == 'token': [B, L, D]
            - Otherwise: [B, D]
        """
        if self.pooling == "token":
            return hidden_states
        
        elif self.pooling == "cls":
            # First token (CLS-style)
            return hidden_states[:, 0, :]
        
        elif self.pooling == "last":
            # Last valid token (common for decoder-only models like Qwen)
            if attention_mask is not None:
                # Find last valid position for each sample
                seq_lens = attention_mask.sum(dim=1)
                batch_size = hidden_states.size(0)
                last_idx = seq_lens - 1
                batch_idx = torch.arange(batch_size, device=hidden_states.device)
                return hidden_states[batch_idx, last_idx.long(), :]
            else:
                return hidden_states[:, -1, :]
        
        elif self.pooling == "mean":
            # Mean pooling over valid tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded.to(hidden_states.dtype), dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
                return sum_embeddings / sum_mask.to(hidden_states.dtype)
            else:
                return hidden_states.mean(dim=1)
        
        else:
            raise ValueError(f"Invalid pooling: {self.pooling}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Qwen3 embeddings for input tokens.
        
        Args:
            input_ids: Input token IDs [B, L]
            attention_mask: Optional attention mask [B, L]
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Embedding tensor:
            - If pooling == 'token': [B, L, output_dim]
            - Otherwise: [B, output_dim]
        """
        device = input_ids.device
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Compute embeddings
        with torch.set_grad_enabled(self.trainable):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
        
        # Apply pooling
        embeddings = self._pool_embeddings(hidden_states, attention_mask)
        
        # Optional output projection
        if self.output_proj is not None:
            embeddings = self.output_proj(embeddings)
        
        return embeddings
    
    def get_output_shape(self, batch_size: int, seq_len: int) -> tuple:
        """
        Get expected output shape for given batch and sequence dimensions.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Expected output shape tuple
        """
        if self.pooling == "token":
            return (batch_size, seq_len, self.output_dim)
        else:
            return (batch_size, self.output_dim)
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer associated with this model."""
        return AutoTokenizer.from_pretrained(
            self.pretrained_model_name,
            trust_remote_code=True,
        )
    
    def extra_repr(self) -> str:
        """String representation of provider configuration."""
        return (
            f"pretrained_model_name={self.pretrained_model_name}, "
            f"hidden_size={self.hidden_size}, "
            f"output_dim={self.output_dim}, "
            f"pooling={self.pooling}, "
            f"trainable={self.trainable}"
        )


__all__ = ["Qwen3EmbeddingProvider"]
