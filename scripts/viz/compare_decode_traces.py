#!/usr/bin/env python3
"""Side-by-side decode video from two traces (e.g. masked vs uniform).

Example::

  python scripts/viz/compare_decode_traces.py \\
    --left docs/research/decode_viz/baseline_masked \\
    --right docs/research/decode_viz/baseline_uniform \\
    --out docs/research/decode_viz/compare_masked_vs_uniform
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
    annotate_intensities,
    ids_to_pieces,
    load_trace,
    visible_span,
)
from render_decode_video import _font, _load_tokenizer, render_frame  # noqa: E402


def _subsample(steps, max_frames: int, stride: int):
  if stride > 1:
    steps = steps[::stride]
  if len(steps) > max_frames:
    idx = [int(i * (len(steps) - 1) / (max_frames - 1))
           for i in range(max_frames)]
    steps = [steps[i] for i in idx]
  return steps


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--left', required=True, type=Path)
  ap.add_argument('--right', required=True, type=Path)
  ap.add_argument('--out', required=True, type=Path)
  ap.add_argument('--max-frames', type=int, default=180)
  ap.add_argument('--stride', type=int, default=1)
  ap.add_argument('--fps', type=int, default=8)
  ap.add_argument('--panel-width', type=int, default=720)
  ap.add_argument('--panel-height', type=int, default=480)
  args = ap.parse_args()

  left = load_trace(args.left)
  right = load_trace(args.right)
  l_steps = _subsample(left['steps'], args.max_frames, args.stride)
  r_steps = _subsample(right['steps'], args.max_frames, args.stride)
  n = max(len(l_steps), len(r_steps))
  # pad by repeating last
  while len(l_steps) < n:
    l_steps.append(l_steps[-1])
  while len(r_steps) < n:
    r_steps.append(r_steps[-1])

  out = Path(args.out)
  frames_dir = out / 'frames'
  frames_dir.mkdir(parents=True, exist_ok=True)
  paths = []
  for i in range(n):
    for side, trace, steps in (
        ('L', left, l_steps),
        ('R', right, r_steps),
    ):
      step = steps[i]
      need_tok = 'token_strs' not in step
      tok = _load_tokenizer(trace) if need_tok else None
      pieces = ids_to_pieces(
          tok, step['token_ids'],
          mask_id=trace.get('mask_id'),
          prefix_len=int(trace.get('prefix_len', 0)),
          token_strs=step.get('token_strs'))
      pieces = annotate_intensities(pieces, step)
      pieces = visible_span(pieces, prefix_len=int(trace.get('prefix_len', 0)))
      panel = render_frame(
          title=str(trace.get('label') or side),
          subtitle=(
              f"{trace.get('forward_process')} · "
              f"{trace.get('decode_profile')}"),
          pieces=pieces,
          step_idx=i,
          n_steps=n,
          window=(int(step['window_start']), int(step['window_end'])),
          width=args.panel_width,
          height=args.panel_height,
      )
      if side == 'L':
        left_img = panel
      else:
        right_img = panel
    gap = 16
    canvas = Image.new(
        'RGB',
        (args.panel_width * 2 + gap + 32, args.panel_height + 48),
        (255, 255, 255))
    canvas.paste(left_img, (16, 32))
    canvas.paste(right_img, (16 + args.panel_width + gap, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 6),
        f"{left.get('label')}  vs  {right.get('label')}",
        fill=(20, 30, 45), font=_font(20))
    path = frames_dir / f'frame_{i:04d}.png'
    canvas.save(path)
    paths.append(path)

  if paths:
    imgs = [Image.open(p) for p in paths]
    gif = out / 'preview.gif'
    imgs[0].save(
        gif, save_all=True, append_images=imgs[1:],
        duration=max(1, int(1000 / args.fps)), loop=0)
    print('wrote', gif)

  ffmpeg = shutil.which('ffmpeg')
  if ffmpeg and paths:
    mp4 = out / 'compare.mp4'
    cmd = [
        ffmpeg, '-y', '-framerate', str(args.fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(mp4),
    ]
    try:
      subprocess.run(cmd, check=True, capture_output=True)
      print('wrote', mp4)
    except subprocess.CalledProcessError as e:
      print('ffmpeg failed', e.stderr.decode()[:300], file=sys.stderr)

  (out / 'compare_meta.json').write_text(json.dumps({
      'left': str(args.left),
      'right': str(args.right),
      'n_frames': len(paths),
  }, indent=2) + '\n')
  print(json.dumps({'out': str(out), 'n_frames': len(paths)}, indent=2))


if __name__ == '__main__':
  main()
