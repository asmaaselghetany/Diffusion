"""Script to generate samples from a trained checkpoint.

This script loads a trained model checkpoint and generates samples using the
configured sampler. The output is saved as a PyTorch tensor (.pt) which can be
used for evaluation (e.g. with generative_ppl.py).
"""

import hydra
import torch
import tqdm
from pathlib import Path

from discrete_diffusion.evaluations.checkpoint_utils import load_block_trainer_checkpoint

@hydra.main(config_path="../../../configs/eval", config_name="generate_samples", version_base="1.3")
def main(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Loading checkpoint from {cfg.checkpoint_path}")
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint_path)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print("Loading tokenizer and model...")
    model, model_config, tokenizer = load_block_trainer_checkpoint(
        checkpoint_path, device)
    print(f"Detected algorithm class: {model.__class__.__name__}")
    
    if cfg.torch_compile:
        print("Compiling model...")
        model = torch.compile(model)

    num_samples = cfg.num_samples
    batch_size = cfg.batch_size
    num_steps = cfg.num_steps
    
    print(f"Generating {num_samples} samples (batch_size={batch_size}, steps={num_steps or 'default'})")
    
    all_samples = []
    
    # Progress bar
    with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Generate samples
            # We use model.generate_samples which delegates to the configured sampler
            samples = model.generate_samples(
                num_samples=current_batch_size,
                num_steps=num_steps
            )
            
            all_samples.append(samples.detach().cpu())
            pbar.update(current_batch_size)
            
    all_samples = torch.cat(all_samples, dim=0)
    
    # Save samples
    out_path = Path(hydra.utils.to_absolute_path(cfg.samples_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(all_samples, out_path)
    print(f"Saved {len(all_samples)} samples to {out_path}")

    if cfg.get("save_text", False):
        print("Decoding samples to text...")
        from discrete_diffusion.data.tokenizers import Text8Tokenizer
        if isinstance(tokenizer, Text8Tokenizer):
            texts = [
                tokenizer.decode_ids_to_text(s, skip_special_tokens=True)
                for s in all_samples]
        else:
            texts = tokenizer.batch_decode(all_samples, skip_special_tokens=True)
        text_path = out_path.with_suffix('.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                f.write(f"Sample {i}:\n{text}\n{'-'*80}\n")
        print(f"Saved text samples to {text_path}")

if __name__ == "__main__":
    main()
