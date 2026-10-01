# Test 3 code-version log (utils.py)

**Date:** 2026-10-01

## Restored / committed blob
- **Path:** `src/discrete_diffusion/forward_process/utils.py`
- **SHA256:** `1a3f6c494ed8bee8ea63fb66d7ff2b3e3f00cb3bf13555a3cb877749d1202f9e`
- **Lines:** 276
- **Source:** `git stash@{0}` (`pre-authorship-rewrite`) — bit-identical to post-restore worktree
- **vs pre-commit HEAD (`89da382`):** was a 45-line stub without `normalize_uniform_*` / `V_eff` helpers (only prior git history on this path: `003caf2` initial). Train/eval on this cluster used the **uncommitted full** file until today's commit.

## Which runs used which code
| Run | Time (CEST) | utils.py |
|-----|-------------|----------|
| `smoke_login_n64` | 13:24 | full (pre-truncate) |
| `probe_check_n64` | 13:33 | full (pre-truncate) |
| `mini_retention` | ~13:40 | full (pre-truncate) |
| Truncation event | ~13:44 | worktree → 45-line stub |
| Failed n=1000 restart | ~13:44 | stub → ImportError (discarded) |
| Restore from stash | ~13:46 | full again (= stash SHA above) |
| `self-check` after restore | 13:48+ | full |
| Current `full_login_n1000` | 13:46+ | full |

## Train-era provenance (honest)

| Check | Result |
|-------|--------|
| File mtime vs job start | **Inconclusive** — current mtime is restore (2026-10-01 13:47); train jobs `2020048`/`2020049` started **2026-09-26 03:40**. mtime cannot prove Sep-26 bytes. |
| Wandb code artifact | **None** — `tmp/code` empty; metadata has `codePath=null`, `git=null`. Logged config only. |
| Slurm / env path | **submit_dir=`Diffusion-new`**, `PYTHONPATH=${REPO_ROOT}/src`, python=`Diffusion/.venv`. Editable resolve today → this tree's `utils.py`. |
| Import / train success | **Strong** — `BlockUniformForwardProcess` imports `normalize_uniform_simplex_mode` at module load; job COMPLETED with that class in the hydra tree. HEAD stub lacks the symbol → **train-time disk had the helpers**. |
| `V_eff` / `normalize_uniform_*` in train log | **Not logged** — no string hit in `slurm_logs/*2020049*.out`. Probe `V_eff=151663` is decode-time only (circumstantial). |
| Byte-identical to stash | **Not proven** for Sep-26. Proven: train needed full API; today's blob is the only surviving full copy (stash + restore). |

## Other trees
| Path | utils.py |
|------|----------|
| `Diffusion-codex-fixes/.../utils.py` | **45-line stub** (`01e0c00d…`) — do not submit from a clean checkout of that tree |
| `Diffusion/.../utils.py` | missing (venv only; package resolves to Diffusion-new) |

## Action
- Commit full `utils.py` + tag baseline; record SHA in table/docs headers.
- Startup assert + SHA in every eval JSON via `code_fingerprint.py`.
- Before booster submit: confirm cluster worktree/`PYTHONPATH` utils is the full blob (not a fresh clone of stub HEAD).
