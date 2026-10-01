#!/usr/bin/env python3
"""Render a Set-Diffusion-style highlight video from a decode trace.

Produces:
  OUT/frames/frame_XXXX.png
  OUT/preview.gif          (always; Pillow)
  OUT/viewer.html          (step scrubber; no ffmpeg needed)
  OUT/decode.mp4           (if ffmpeg is on PATH)

Example::

  python scripts/viz/render_decode_video.py \\
    --trace docs/research/decode_viz/baseline_masked \\
    --out docs/research/decode_viz/baseline_masked/render
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / 'src'))

from decode_trace_lib import (  # noqa: E402
    GOLD_RGB,
    annotate_intensities,
    ids_to_pieces,
    load_trace,
    visible_span,
)
from transformers import AutoTokenizer  # noqa: E402


def _font(size: int) -> ImageFont.ImageFont:
  for name in (
      '/usr/share/fonts/dejavu/DejaVuSansMono.ttf',
      '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
      '/usr/share/fonts/liberation/LiberationMono-Regular.ttf',
  ):
    if Path(name).is_file():
      return ImageFont.truetype(name, size=size)
  return ImageFont.load_default()


def _wrap_pieces(
    pieces: list[dict],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
    x0: int,
) -> list[tuple[dict, int, int]]:
  """Return (piece, x, y) layout with simple wrapping."""
  x, y = x0, 0
  line_h = font.size + 6
  laid = []
  for p in pieces:
    text = p.get('text') or ''
    if text == '':
      continue
    # handle embedded newlines
    parts = text.split('\n')
    for j, part in enumerate(parts):
      if j > 0:
        x = x0
        y += line_h
      if part == '':
        continue
      bbox = draw.textbbox((0, 0), part, font=font)
      w = bbox[2] - bbox[0]
      if x > x0 and x + w > max_width:
        x = x0
        y += line_h
      laid.append((dict(p, text=part), x, y))
      x += w
  return laid


def render_frame(
    *,
    title: str,
    subtitle: str,
    pieces: list[dict],
    step_idx: int,
    n_steps: int,
    window: tuple[int, int],
    width: int = 960,
    height: int = 540,
) -> Image.Image:
  img = Image.new('RGB', (width, height), (255, 255, 255))
  draw = ImageDraw.Draw(img)
  title_font = _font(22)
  body_font = _font(18)
  small = _font(14)

  draw.text((24, 16), title, fill=(20, 30, 45), font=title_font)
  draw.text((24, 46), subtitle, fill=(90, 100, 120), font=small)
  draw.text(
      (24, 68),
      f'step {step_idx + 1}/{n_steps}   window [{window[0]}, {window[1]})',
      fill=(90, 100, 120), font=small)

  # legend
  lx = width - 280
  draw.text((lx, 16), 'highlight =', fill=(90, 100, 120), font=small)
  labels = ['active / new', '', '', 'pending mask']
  for i, rgb in enumerate(GOLD_RGB):
    bx = lx + i * 36
    draw.rectangle([bx, 36, bx + 28, 54], fill=rgb, outline=(180, 150, 40))
  draw.text((lx, 58), 'more likely → less', fill=(120, 110, 60), font=small)

  # body
  canvas = Image.new('RGB', (width - 48, height - 120), (252, 253, 255))
  cdraw = ImageDraw.Draw(canvas)
  laid = _wrap_pieces(pieces, cdraw, body_font, max_width=width - 80, x0=8)
  for p, x, y in laid:
    text = p['text']
    inten = p.get('intensity')
    bbox = cdraw.textbbox((x, y), text, font=body_font)
    if inten is not None:
      pad = 1
      cdraw.rectangle(
          [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
          fill=GOLD_RGB[min(inten, 3)])
    color = (40, 45, 55)
    if p.get('role') == 'mask':
      color = (160, 150, 100)
    elif p.get('role') == 'prefix':
      color = (70, 80, 95)
    cdraw.text((x, y), text, fill=color, font=body_font)
  img.paste(canvas, (24, 96))
  # footer line
  draw.rectangle([0, height - 4, width, height], fill=(31, 122, 79))
  return img


def _load_tokenizer(trace: dict):
  # Prefer tokenizer from checkpoint config if present; else Qwen Instruct.
  ckpt = trace.get('checkpoint')
  if ckpt:
    try:
      import torch
      from omegaconf import OmegaConf
      from discrete_diffusion.data import get_tokenizer
      blob = torch.load(ckpt, map_location='cpu', weights_only=False)
      cfg = blob['hyper_parameters']['config']
      if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
      return get_tokenizer(cfg)
    except Exception:
      pass
  try:
    return AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-1.5B-Instruct', use_fast=True, local_files_only=True)
  except Exception:
    class _Dummy:
      def decode(self, ids, skip_special_tokens=False):
        return ''.join(f'<{t}>' for t in ids)
    return _Dummy()



def write_viewer_html(out_dir: Path, n_frames: int, title: str) -> None:
  html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#f6f8fb; margin:0; }}
main {{ max-width: 1000px; margin: 1rem auto; padding: 0 1rem; }}
img {{ width: 100%; background:#fff; border:1px solid #dfe5ee; border-radius:8px; }}
input {{ width: 100%; }}
</style></head><body><main>
<h2>{title}</h2>
<img id="f" src="frames/frame_0000.png" alt="frame">
<input id="s" type="range" min="0" max="{n_frames - 1}" value="0">
<p><span id="lab">0</span> / {n_frames - 1}
<button id="play">Play</button></p>
<script>
const s=document.getElementById('s'), f=document.getElementById('f'),
 lab=document.getElementById('lab'), play=document.getElementById('play');
let t=null;
function show(i){{
  f.src='frames/frame_'+String(i).padStart(4,'0')+'.png';
  lab.textContent=i; s.value=i;
}}
s.oninput=()=>show(+s.value);
play.onclick=()=>{{
  if(t){{clearInterval(t); t=null; play.textContent='Play'; return;}}
  play.textContent='Pause';
  t=setInterval(()=>{{
    let i=(+s.value+1)%{n_frames}; show(i);
  }}, 120);
}};
</script></main></body></html>
"""
  (out_dir / 'viewer.html').write_text(html)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--trace', required=True, type=Path)
  ap.add_argument('--out', required=True, type=Path)
  ap.add_argument('--stride', type=int, default=1, help='Keep every Nth event')
  ap.add_argument('--max-frames', type=int, default=240)
  ap.add_argument('--fps', type=int, default=8)
  ap.add_argument('--width', type=int, default=960)
  ap.add_argument('--height', type=int, default=540)
  args = ap.parse_args()

  trace = load_trace(args.trace)
  steps = trace['steps']
  if args.stride > 1:
    steps = steps[:: args.stride]
  if len(steps) > args.max_frames:
    # uniform subsample
    idx = [int(i * (len(steps) - 1) / (args.max_frames - 1))
           for i in range(args.max_frames)]
    steps = [steps[i] for i in idx]

  mask_id = trace.get('mask_id')
  prefix_len = int(trace.get('prefix_len', 0))
  label = trace.get('label') or 'decode'
  title = f"{label}"
  subtitle = (
      f"{trace.get('forward_process', '')} · profile={trace.get('decode_profile')} "
      f"· thr={trace.get('unmask_threshold')}")

  # Prefer pre-decoded token_strs (offline). Fall back to ckpt tokenizer.
  need_tok = not all('token_strs' in s for s in steps)
  tok = _load_tokenizer(trace) if need_tok else None

  out = Path(args.out)
  frames_dir = out / 'frames'
  frames_dir.mkdir(parents=True, exist_ok=True)
  paths = []
  for fi, step in enumerate(steps):
    pieces = ids_to_pieces(
        tok, step['token_ids'], mask_id=mask_id, prefix_len=prefix_len,
        token_strs=step.get('token_strs'))
    pieces = annotate_intensities(pieces, step)
    pieces = visible_span(pieces, prefix_len=prefix_len)
    img = render_frame(
        title=title,
        subtitle=subtitle,
        pieces=pieces,
        step_idx=fi,
        n_steps=len(steps),
        window=(int(step['window_start']), int(step['window_end'])),
        width=args.width,
        height=args.height,
    )
    path = frames_dir / f'frame_{fi:04d}.png'
    img.save(path)
    paths.append(path)

  # GIF
  if paths:
    imgs = [Image.open(p) for p in paths]
    gif = out / 'preview.gif'
    imgs[0].save(
        gif, save_all=True, append_images=imgs[1:],
        duration=max(1, int(1000 / args.fps)), loop=0)
    print('wrote', gif)

  write_viewer_html(out, len(paths), title)

  # MP4 if ffmpeg exists
  ffmpeg = shutil.which('ffmpeg')
  if ffmpeg and paths:
    mp4 = out / 'decode.mp4'
    cmd = [
        ffmpeg, '-y', '-framerate', str(args.fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(mp4),
    ]
    try:
      subprocess.run(cmd, check=True, capture_output=True)
      print('wrote', mp4)
    except subprocess.CalledProcessError as e:
      print('ffmpeg failed:', e.stderr.decode()[:400], file=sys.stderr)
  else:
    print('ffmpeg not found — use preview.gif or viewer.html')

  (out / 'render_meta.json').write_text(json.dumps({
      'n_frames': len(paths),
      'stride': args.stride,
      'fps': args.fps,
      'source_events': trace.get('n_events'),
  }, indent=2) + '\n')
  print(json.dumps({'out': str(out), 'n_frames': len(paths)}, indent=2))


if __name__ == '__main__':
  main()
