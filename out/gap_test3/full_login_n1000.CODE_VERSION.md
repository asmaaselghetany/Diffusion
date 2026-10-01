# Sidecar: full_login_n1000 code version

The result JSON `full_login_n1000.json` was started **before**
`code_fingerprint` was wired into `gap_test3_denoiser_quality.py`, so that
field will be absent. This sidecar is the record.

| Field | Value |
|-------|--------|
| Process PID | `3898104` (wrapper `3898078`) |
| Host | login node (CUDA_VISIBLE_DEVICES=0) |
| Command start (CEST) | **2026-10-01 13:46:55** |
| Log mtime start | 2026-10-01 13:55:05 (first log lines) |
| utils.py restore | ~**2026-10-01 13:46** from `stash@{0}` |
| utils.py mtime at restore | 2026-10-01 13:47:01 |
| Truncation event | ~13:44 → stub; discarded restart hit ImportError |
| SHA256 of `forward_process/utils.py` during this run | `1a3f6c494ed8bee8ea63fb66d7ff2b3e3f00cb3bf13555a3cb877749d1202f9e` |
| Commit that recorded that blob | `29e3ca6` (tag `baseline-ar2block-utils-full-2026-10-01`) — committed **after** this job started; worktree bytes already matched |
| Output | `out/gap_test3/full_login_n1000.json` |

**Ordering proof:** start (13:46:55) is after restore (~13:46) and after the
failed stub restart. Post-restore `self-check` passed before/alongside this
job. Do not treat missing `code_fingerprint` in the JSON as unknown code —
use this sidecar.
