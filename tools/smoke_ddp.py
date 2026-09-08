#!/usr/bin/env python3
"""Probe GPU visibility under a *single* Slurm task (legacy Lightning spawn).

Under SlurmEnvironment, ``Fabric(devices=N)`` from ``srun --ntasks=1`` does
*not* spawn sibling ranks — world_size stays 1. The real multi-GPU path is
external ranks via ``scripts/_block_qwen_ddp.bash`` (one task per GPU).
"""

from __future__ import annotations

import os

import lightning as L
import torch


def _worker(fabric: L.Fabric) -> None:
  rank = fabric.global_rank
  world = fabric.world_size
  dev = torch.cuda.current_device()
  name = torch.cuda.get_device_name(dev)
  fabric.barrier()
  if rank == 0:
    print(
        f"smoke_ddp (legacy ntasks=1): world_size={world} device={dev} ({name})",
        flush=True,
    )
    if world == 1:
      print(
          "smoke_ddp NOTE: world_size=1 confirms SlurmEnvironment does not "
          "spawn ranks from a lone task — use external-rank launch.",
          flush=True,
      )


def main() -> None:
  if not torch.cuda.is_available():
    raise SystemExit("CUDA required")
  n = torch.cuda.device_count()
  if n < 2:
    raise SystemExit(f"Need >=2 visible GPUs, saw {n}")
  devices = min(4, n)
  master = os.environ.get("MASTER_ADDR", "127.0.0.1")
  port = os.environ.get("MASTER_PORT", "29500")
  print(f"smoke_ddp: requesting devices={devices} master={master}:{port}",
        flush=True)
  fabric = L.Fabric(devices=devices, accelerator="cuda")
  fabric.launch(_worker)


if __name__ == "__main__":
  main()
