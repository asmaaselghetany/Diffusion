# Run artifacts live outside this git tree

Heavy checkpoints / Slurm logs:  
`/fast/project/HFMI_SynergyUnit/asmaa.elsayed/uni-d2-data/`

See that README for `paper_arms/` vs `micros/`.

`outputs/block_qwen/<run>` and `slurm_logs/` are **symlinks** into `uni-d2-data` (except any in-progress job still writing under `outputs/`).
