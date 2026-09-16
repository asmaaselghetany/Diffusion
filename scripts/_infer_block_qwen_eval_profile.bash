#!/usr/bin/env bash
# Infer eval family / decode profile from a block_qwen run (hydra config next
# to the checkpoint — avoids loading multi‑GB weights).
#
# Shared default for fair cross-pipeline compare:
#   DECODE_PROFILE=baseline → BlockSampler ancestral (steps=32, no ARPC,
#   no unmask_threshold, no DualCache). Cleared even if the ckpt was trained
#   with Fast-dLLM/BlockGen paper pins.
#
# Exactness overlays (recipe-specific):
#   fastdllm_lm_eval  → dual_cache + thr=1 (paper accuracy) + greedy
#                       (Hub/speed overlay: FORCE_UNMASK_THRESHOLD=0.9)
#   blockgen_arpc     → offline ELBO/gen-PPL + ARPC samples

infer_block_qwen_eval_profile() {
  local ckpt="${1:?ckpt}"
  local ckpt_abs
  ckpt_abs="$(readlink -f "${ckpt}")"
  local run_dir
  run_dir="$(dirname "$(dirname "${ckpt_abs}")")"
  local cfg=""
  for cand in \
    "${run_dir}/hydra/.hydra/config.yaml" \
    "${run_dir}/.hydra/config.yaml" \
    "$(dirname "${ckpt_abs}")/../hydra/.hydra/config.yaml"
  do
    if [[ -f "${cand}" ]]; then
      cfg="$(readlink -f "${cand}")"
      break
    fi
  done
  if [[ -z "${cfg}" ]]; then
    echo "infer_block_qwen_eval_profile: no hydra config near ${ckpt_abs}" >&2
    return 2
  fi

  local py="${REPO_ROOT:-.}/.venv/bin/python"
  [[ -x "${py}" ]] || py=python

  eval "$(
    "${py}" - "${cfg}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
fp = OmegaConf.select(cfg, "algo.forward_process_name")
if not fp:
    name = OmegaConf.select(cfg, "algo.forward_process.name")
    if name:
        fp = str(name).replace("block_", "")
    else:
        target = str(OmegaConf.select(cfg, "algo.forward_process._target_") or "")
        if "masked" in target:
            fp = "masked"
        elif "uniform" in target:
            fp = "uniform"
        elif "hybrid" in target:
            fp = "hybrid"
        else:
            fp = "unknown"

use_arpc = bool(OmegaConf.select(cfg, "sampling.use_arpc") or False)
arpc_mode = OmegaConf.select(cfg, "sampling.arpc_mode")
thr = OmegaConf.select(cfg, "sampling.unmask_threshold")
greedy = OmegaConf.select(cfg, "sampling.greedy")
hier = bool(OmegaConf.select(cfg, "sampling.hierarchical_kv") or False)
dual = bool(OmegaConf.select(cfg, "sampling.use_block_cache") or False)
single = bool(OmegaConf.select(cfg, "sampling.single_stream_decode") or False)
line = OmegaConf.select(cfg, "line") or OmegaConf.select(cfg, "experiment.line")

exact = (
    fp == "masked"
    and thr is not None
    and float(thr) > 0
    and bool(greedy)
    and hier
    and dual
    and single
)

if fp == "uniform" or use_arpc:
    stack = "blockgen_arpc"
    family = "native" if (not line or str(line) == "block") else "transfer"
    decode = "arpc_blockgen"
    reason = "uniform/ARPC → offline ELBO+gen-PPL + arpc_decode_eval (not chat lm-eval)"
elif exact:
    stack = "fastdllm_lm_eval"
    family = "conversion"
    decode = "dual_cache"
    reason = "masked + Hub-ish confidence/DualCache pins → chat lm-eval exactness"
elif fp == "masked":
    stack = "conversion_lm_eval"
    family = "conversion"
    decode = "baseline"
    reason = "masked without full DualCache pins → chat lm-eval baseline (C0-style floor)"
else:
    stack = "unknown"
    family = "unknown"
    decode = "baseline"
    reason = f"unrecognized forward_process={fp!r}"

def sh(v):
    if v is None:
        return ""
    return str(v).replace("'", "'\"'\"'")

print(f"EVAL_FORWARD='{sh(fp)}'")
print(f"EVAL_FAMILY='{sh(family)}'")
print(f"EVAL_STACK='{sh(stack)}'")
print(f"EVAL_DECODE_PROFILE='{sh(decode)}'")
print(f"EVAL_UNMASK_THRESHOLD='{sh(thr) if thr is not None else ''}'")
print(f"EVAL_FORCE_GREEDY='{'1' if greedy else ''}'")
print(f"EVAL_USE_ARPC='{'1' if use_arpc else '0'}'")
print(f"EVAL_ARPC_MODE='{sh(arpc_mode) if arpc_mode else 'blockgen'}'")
print(f"EVAL_REASON='{sh(reason)}'")
print(f"EVAL_HYDRA_CFG='{sh(sys.argv[1])}'")
PY
  )"

  export EVAL_FORWARD EVAL_FAMILY EVAL_STACK EVAL_DECODE_PROFILE
  export EVAL_UNMASK_THRESHOLD EVAL_FORCE_GREEDY EVAL_USE_ARPC EVAL_ARPC_MODE
  export EVAL_REASON EVAL_HYDRA_CFG
}
