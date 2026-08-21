#!/usr/bin/env bash
# Shared helpers for block_qwen checkpoint safety.
# NEVER hardlink or symlink checkpoints — a later save can clobber every alias.

# True if path is a readable non-empty zip (Lightning .ckpt).
_block_qwen_ckpt_ok() {
  local path="$1"
  [[ -f "${path}" && ! -L "${path}" ]] || return 1
  [[ "$(stat -c%s "${path}" 2>/dev/null || echo 0)" -gt 1000000 ]] || return 1
  python - "${path}" <<'PY'
import sys, zipfile
p = sys.argv[1]
try:
  zipfile.ZipFile(p)
except Exception:
  raise SystemExit(1)
raise SystemExit(0)
PY
}

# Print free GiB on the filesystem that holds $1.
_block_qwen_free_gb() {
  df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

# Fail if fewer than MIN_GB free under path (default 80).
_block_qwen_require_disk() {
  local path="$1"
  local min_gb="${2:-${BLOCK_QWEN_MIN_FREE_GB:-80}}"
  local free
  free="$(_block_qwen_free_gb "${path}")"
  if [[ -z "${free}" ]]; then
    echo "WARNING: could not determine free disk for ${path}" >&2
    return 0
  fi
  if (( free < min_gb )); then
    echo "ERROR: only ${free}G free under ${path}; need >= ${min_gb}G before checkpoint writes." >&2
    echo "  Free space or set BLOCK_QWEN_MIN_FREE_GB lower (not recommended)." >&2
    return 1
  fi
  echo "disk_ok: ${free}G free (min ${min_gb}G) on $(df -h "${path}" | tail -1 | awk '{print $1,$6}')"
}

# Keep at most KEEP newest valid periodic ckpts (0-*.ckpt and N-*.ckpt), default 2.
# Never touches last*.ckpt / best.ckpt. Never uses links.
_block_qwen_prune_periodics() {
  local ckpt_dir="$1"
  local keep="${2:-${BLOCK_QWEN_KEEP_PERIODICS:-2}}"
  [[ -d "${ckpt_dir}" ]] || return 0
  mapfile -t cands < <(
    find "${ckpt_dir}" -maxdepth 1 -type f \( -name '0-*.ckpt' -o -name '[0-9]-*.ckpt' \) \
      ! -name 'last*.ckpt' ! -name 'best.ckpt' -printf '%f\t%T@\t%p\n' 2>/dev/null \
      | sort -t$'\t' -k1,1Vr -k2,2nr
  )
  local i=0
  local line cand
  for line in "${cands[@]:-}"; do
    [[ -n "${line}" ]] || continue
    cand="${line##*$'\t'}"
    [[ -n "${cand}" && -e "${cand}" ]] || continue
    if ! _block_qwen_ckpt_ok "${cand}"; then
      echo "ckpt: removing invalid periodic $(basename "${cand}")" >&2
      rm -f "${cand}"
      continue
    fi
    i=$((i + 1))
    if (( i > keep )); then
      echo "ckpt: pruning old periodic $(basename "${cand}") (keep=${keep})" >&2
      rm -f "${cand}"
    fi
  done
}

# Print global_step for a Lightning .ckpt (or -1 on failure).
_block_qwen_ckpt_global_step() {
  local path="$1"
  python - "${path}" <<'PY'
import sys, gc
import torch
p = sys.argv[1]
try:
  obj = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
  print(int(obj.get("global_step") or -1))
  del obj
  gc.collect()
except Exception:
  raise SystemExit(1)
PY
}

# Resolve trainer.max_steps from env / HYDRA_OVERRIDES (default 6000).
_block_qwen_max_steps() {
  local max_steps="${BLOCK_QWEN_MAX_STEPS:-6000}"
  if [[ "${HYDRA_OVERRIDES:-}" =~ trainer\.max_steps=([0-9]+) ]]; then
    max_steps="${BASH_REMATCH[1]}"
  fi
  echo "${max_steps}"
}

# Print ``<abs_path>\t<global_step>`` for the highest-step valid ckpt (no copy).
# Periodics use the step in ``0-N.ckpt``; last/best are torch-loaded.
_block_qwen_pick_highest_ckpt() {
  local ckpt_dir="$1"
  # Silence transformers FutureWarning etc. so stdout stays machine-parseable.
  PYTHONWARNINGS=ignore python - "${ckpt_dir}" <<'PY'
import re, sys, zipfile, gc, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
cands = []
# Lightning ModelCheckpoint default: ``{epoch}-{step}.ckpt`` (epoch may be 0 or 1+).
step_re = re.compile(r'^\d+-(\d+)\.ckpt$')
for p in ckpt_dir.glob('*.ckpt'):
  if p.is_symlink():
    continue
  try:
    zipfile.ZipFile(p)
  except Exception:
    continue
  name = p.name
  m = step_re.match(name)
  if m:
    step = int(m.group(1))
  else:
    try:
      import torch
      obj = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
      step = int(obj.get('global_step') or -1)
      del obj
      gc.collect()
    except Exception:
      continue
  if name == 'last.ckpt':
    prio = 3
  elif name.startswith('last'):
    prio = 2
  elif name == 'best.ckpt':
    prio = 1
  else:
    prio = 0
  cands.append((step, prio, str(p.resolve())))

if not cands:
  raise SystemExit(1)
cands.sort(reverse=True)
print(f"{cands[0][2]}\t{cands[0][0]}")
PY
}

# Optional: materialize highest ckpt as last.ckpt via cp (never link).
# Prefer pick + checkpointing.resume_ckpt_path override to avoid Lustre copies.
_block_qwen_prepare_last_ckpt() {
  local ckpt_dir="$1"
  local last="${ckpt_dir}/last.ckpt"
  mkdir -p "${ckpt_dir}"
  _block_qwen_prune_periodics "${ckpt_dir}"

  local src pick
  pick="$(_block_qwen_pick_highest_ckpt "${ckpt_dir}")" || {
    echo "ERROR: no valid checkpoint in ${ckpt_dir}" >&2
    return 1
  }
  src="${pick%%$'\t'*}"

  if [[ "${src}" == "$(readlink -f "${last}" 2>/dev/null || echo "${last}")" ]] \
      || [[ "${src}" == "${last}" ]]; then
    echo "ckpt: using valid last.ckpt (highest step)" >&2
    echo "${last}"
    return 0
  fi

  if [[ -e "${last}" || -L "${last}" ]]; then
    echo "ckpt: replacing stale/invalid last.ckpt with $(basename "${src}")" >&2
    rm -f "${last}"
  else
    echo "ckpt: creating last.ckpt via cp from $(basename "${src}") (never link)" >&2
  fi
  cp -f "${src}" "${last}"
  if ! _block_qwen_ckpt_ok "${last}"; then
    echo "ERROR: copy to last.ckpt failed validation" >&2
    return 1
  fi
  echo "${last}"
}
