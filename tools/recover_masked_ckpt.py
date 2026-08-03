#!/usr/bin/env python3
import shutil
import zipfile
from pathlib import Path

d = Path("outputs/block_qwen/ar2block_masked_131655/checkpoints")
src = d / "0-3000.ckpt"
zipfile.ZipFile(src)
print("src OK", src, f"{src.stat().st_size/1e9:.2f}G")

for name in ["last.ckpt", "best.ckpt"]:
  dst = d / name
  if dst.exists():
    q = d / f"{name}.quarantine_{dst.stat().st_mtime_ns}"
    dst.rename(q)
    print("quarantined", dst, "->", q.name)
  shutil.copy2(src, dst)
  assert dst.stat().st_nlink == 1, dst.stat().st_nlink
  zipfile.ZipFile(dst)
  print("OK", name, f"{dst.stat().st_size/1e9:.2f}G", "inode", dst.stat().st_ino)
