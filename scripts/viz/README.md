# Decode highlight videos (Set-Diffusion-style)

Offline pipeline to visualize **how** our block models decode over time.
Reusable for every trial (masked / uniform / hybrid, any decode profile).

## What you get

| Artifact | Role |
|----------|------|
| `meta.json` + `events.jsonl` | Per-step trace: window, active/new positions, token ids |
| `render/preview.gif` | Animated highlight (no ffmpeg needed) |
| `render/viewer.html` | Scrub frames in a browser |
| `render/decode.mp4` | Only if `ffmpeg` is on `PATH` |
| compare `preview.gif` | Side-by-side two trials |

Gold highlight = tokens being decoded in the current block window (bright = newly committed; dim = still masked / pending).

## Quick start

```bash
cd /e/project1/scifi/elsayed3/Diffusion-new
export PYTHONPATH=src
source .venv/bin/activate

# Masked baseline (hubmatch thr=1 — paper accuracy decode)
./scripts/viz/make_decode_demo.sh \
  --name baseline_masked \
  --ckpt outputs/block_qwen/ar2block_masked_1762534/checkpoints/last.ckpt \
  --profile hubmatch --thr 1.0

# Uniform baseline (ancestral; thr null)
./scripts/viz/make_decode_demo.sh \
  --name baseline_uniform \
  --ckpt outputs/block_qwen/ar2block_uniform_1955203/checkpoints/last.ckpt \
  --profile baseline

# Side-by-side (corruption story)
./scripts/viz/make_decode_demo.sh --compare \
  docs/research/decode_viz/baseline_masked \
  docs/research/decode_viz/baseline_uniform
```

Custom prompt:

```bash
PROMPT='Your question here' ./scripts/viz/make_decode_demo.sh \
  --name my_run --ckpt /path/to/last.ckpt --profile dual_cache --thr 1.0
```

## Pieces

| Script | Job |
|--------|-----|
| `record_decode_trace.py` | Load ckpt → hook `BlockSampler.step_hook` → dump events |
| `render_decode_video.py` | Trace → PNG frames + GIF + HTML (+ MP4) |
| `compare_decode_traces.py` | Two traces → side-by-side GIF |
| `make_decode_demo.sh` | Thin wrapper for the above |

Instrumentation lives in `BlockSampler._emit_step_hook` (optional; zero cost when `step_hook is None`).

## Notes

- Same GSM-style prompt across trials makes slides fair.
- Uniform free-gen may look like soup — that **is** the point for Claim K / corruption.
- For presentation accuracy decode on masked, prefer `--profile hubmatch --thr 1.0`.
- Walltime: ~minutes on 1 GPU for `max_new=128`, `steps=32`.
