#!/usr/bin/env python3
"""Evaluate continuous Gaussian-latent diffusion model capabilities.

Tasks:
- reconstruction: encoder -> decoder only
- unconditional: Gaussian latent noise -> denoise -> decode
- infill: contiguous-span latent-conditioned infilling
- prompt_qa: prompt-based answer generation from local JSONL prompts
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import omegaconf
from omegaconf import OmegaConf

# Add src to import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from discrete_diffusion.evaluations.continuous_capabilities.checkpoints import (
    CheckpointRecord,
    discover_checkpoint_records,
    load_trainer_from_checkpoint,
)
from discrete_diffusion.evaluations.continuous_capabilities.data import (
    load_fixed_text_records,
    load_qa_prompt_records,
    load_validation_token_samples,
    tokenize_fixed_text_records,
)
from discrete_diffusion.evaluations.continuous_capabilities.metrics import ExternalPPLEvaluator
from discrete_diffusion.evaluations.continuous_capabilities.reporting import (
    write_results_csv,
    write_results_json,
    write_sample_artifacts,
)
from discrete_diffusion.evaluations.continuous_capabilities.tasks import (
    SamplingConfig,
    run_infilling,
    run_prompt_qa,
    run_reconstruction_encode_decode,
    run_unconditional_generation,
)


def _register_resolver(name: str, resolver) -> None:
    if omegaconf.OmegaConf.has_resolver(name):
        return
    omegaconf.OmegaConf.register_new_resolver(name, resolver)


def _register_omegaconf_resolvers() -> None:
    # Mirror training entrypoint resolvers so checkpoint configs with
    # interpolations (e.g., ${mul:...}, ${div_up:...}) can be evaluated.
    _register_resolver("cwd", os.getcwd)
    _register_resolver("device_count", torch.cuda.device_count)
    _register_resolver("div_up", lambda x, y: (x + y - 1) // y)

    def _mul(*args):
        out = 1
        for arg in args:
            out *= arg
        return out

    _register_resolver("mul", _mul)
    _register_resolver("sub", lambda x, y: x - y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate continuous model capabilities")

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["reconstruction", "unconditional", "infill", "prompt_qa", "all"],
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/continuous_embedding",
        help="Root directory containing run subdirectories",
    )
    parser.add_argument(
        "--checkpoint_pattern",
        type=str,
        default="**/checkpoints/best.ckpt",
        help="Glob pattern relative to --checkpoint_dir",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Runtime preset from configs/eval/continuous_capabilities_<preset>.yaml",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim", "ancestral"],
    )
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--adaptive_lm_guidance_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_lm_model", type=str, default="gpt2")
    parser.add_argument(
        "--adaptive_lm_guidance_mode",
        type=str,
        default="confidence",
        choices=["static", "confidence"],
    )
    parser.add_argument("--adaptive_lm_guidance_min_scale", type=float, default=0.0)
    parser.add_argument("--adaptive_lm_guidance_scale_conditioned", type=float, default=None)
    parser.add_argument("--max_decode_tokens", type=int, default=None)
    parser.add_argument(
        "--qa_file",
        type=str,
        default="configs/eval/continuous_qa_prompts.sample.jsonl",
    )
    parser.add_argument(
        "--fixed_texts_file",
        type=str,
        default="configs/eval/continuous_fixed_texts.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reports and artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--max_checkpoints", type=int, default=None)
    parser.add_argument("--save_samples", action="store_true")

    # Optional practical overrides
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--force_stage2_tasks", action="store_true")
    parser.add_argument("--ban_special_tokens", action=argparse.BooleanOptionalAction, default=None)

    return parser.parse_args()



def _load_preset_config(preset_name: str) -> Dict[str, Any]:
    preset_path = Path("configs/eval") / f"continuous_capabilities_{preset_name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset config not found: {preset_path}")
    cfg = OmegaConf.load(preset_path)
    return OmegaConf.to_container(cfg, resolve=True)



def _build_output_dir(explicit_output_dir: str | None, preset: str) -> Path:
    if explicit_output_dir:
        out_dir = Path(explicit_output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs/continuous_embedding/eval/capability") / f"{timestamp}_{preset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _task_list(task_arg: str) -> List[str]:
    if task_arg == "all":
        return ["reconstruction", "unconditional", "infill", "prompt_qa"]
    return [task_arg]



def _skip_stage2_task(task_name: str, is_decoder_only: bool, force_stage2_tasks: bool) -> bool:
    if force_stage2_tasks:
        return False
    if not is_decoder_only:
        return False
    return task_name in {"unconditional", "infill", "prompt_qa"}



def _resolve_sampling_cfg(args: argparse.Namespace, preset: Dict[str, Any]) -> SamplingConfig:
    eta_arg = getattr(args, "eta", None)
    temperature_arg = getattr(args, "temperature", None)
    top_p_arg = getattr(args, "top_p", None)
    top_k_arg = getattr(args, "top_k", None)
    num_steps_arg = getattr(args, "num_steps", None)

    eta = eta_arg if eta_arg is not None else float(preset.get("eta", 1.0))
    temperature = temperature_arg if temperature_arg is not None else float(preset.get("temperature", 1.0))
    top_p = top_p_arg if top_p_arg is not None else float(preset.get("top_p", 1.0))
    top_k = top_k_arg if top_k_arg is not None else int(preset.get("top_k", 0))
    num_steps = num_steps_arg if num_steps_arg is not None else int(preset["num_steps"])

    ban_special_tokens = (
        getattr(args, "ban_special_tokens", None)
        if getattr(args, "ban_special_tokens", None) is not None
        else bool(preset.get("ban_special_tokens", True))
    )

    adaptive_lm_guidance_scale = float(getattr(args, "adaptive_lm_guidance_scale", 0.0))
    adaptive_lm_model = str(getattr(args, "adaptive_lm_model", "gpt2"))
    adaptive_lm_guidance_mode = str(getattr(args, "adaptive_lm_guidance_mode", "confidence"))
    adaptive_lm_guidance_min_scale = float(getattr(args, "adaptive_lm_guidance_min_scale", 0.0))
    adaptive_lm_guidance_scale_conditioned = getattr(args, "adaptive_lm_guidance_scale_conditioned", None)
    max_decode_tokens = getattr(args, "max_decode_tokens", None)

    return SamplingConfig(
        num_steps=num_steps,
        sampling_method=str(getattr(args, "sampling_method", "ddpm")),
        eta=eta,
        eps=float(preset.get("sampling_eps", 1e-5)),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        ban_special_tokens=ban_special_tokens,
        clip_denoised=bool(preset.get("clip_denoised", False)),
        clip_range=tuple(preset.get("clip_range", [-10.0, 10.0])),
        adaptive_lm_guidance_scale=adaptive_lm_guidance_scale,
        adaptive_lm_model_name=adaptive_lm_model,
        adaptive_lm_guidance_mode=adaptive_lm_guidance_mode,
        adaptive_lm_guidance_min_scale=adaptive_lm_guidance_min_scale,
        adaptive_lm_guidance_scale_conditioned=adaptive_lm_guidance_scale_conditioned,
        max_decode_tokens=max_decode_tokens,
    )



def _prepare_sources(
    *,
    model_config,
    tokenizer,
    fixed_texts_file: str,
    num_samples: int,
    device: torch.device,
    seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []

    seq_len = int(model_config.model.length)

    # Validation source
    validation_source, validation_error = load_validation_token_samples(
        model_config,
        tokenizer,
        max_samples=num_samples,
        device=device,
        seed=seed,
    )
    if validation_source is not None:
        sources.append(validation_source)
        metadata["validation_available"] = True
        metadata["validation_samples"] = int(validation_source["input_ids"].shape[0])
    else:
        metadata["validation_available"] = False
        metadata["validation_error"] = validation_error

    # Fixed texts source
    try:
        fixed_records = load_fixed_text_records(fixed_texts_file, max_records=num_samples)
        fixed_source = tokenize_fixed_text_records(
            tokenizer,
            fixed_records,
            max_length=seq_len,
            device=device,
        )
        if int(fixed_source["input_ids"].shape[0]) > 0:
            sources.append(fixed_source)
        metadata["fixed_text_samples"] = int(fixed_source["input_ids"].shape[0])
    except Exception as exc:  # pylint: disable=broad-except
        metadata["fixed_text_samples"] = 0
        metadata["fixed_text_error"] = str(exc)

    return sources, metadata



def _record_summary_line(checkpoint_result: Dict[str, Any]) -> str:
    run_name = checkpoint_result.get("run_name", "unknown")
    if checkpoint_result.get("error"):
        return f"{run_name} -> error: {checkpoint_result['error']}"
    tasks = checkpoint_result.get("tasks", {})
    status = []
    for task_name, task_payload in tasks.items():
        status.append(f"{task_name}:{task_payload.get('status', 'ok')}")
    if not status:
        return f"{run_name} -> no tasks recorded"
    return f"{run_name} -> {' | '.join(status)}"



def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    _register_omegaconf_resolvers()
    preset = _load_preset_config(args.preset)
    output_dir = _build_output_dir(args.output_dir, args.preset)

    num_samples = args.num_samples if args.num_samples is not None else int(preset["num_samples"])
    reconstruction_batch_size = int(preset.get("reconstruction_batch_size", 8))
    generation_batch_size = int(preset.get("generation_batch_size", 8))
    sample_save_limit = int(preset.get("sample_save_limit", 32))

    sampling_cfg = _resolve_sampling_cfg(args, preset)
    task_names = _task_list(args.task)

    _set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    records = discover_checkpoint_records(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_pattern=args.checkpoint_pattern,
        max_checkpoints=args.max_checkpoints,
        continuous_only=True,
    )

    if not records:
        raise RuntimeError(
            "No continuous checkpoints found under "
            f"{args.checkpoint_dir}/{args.checkpoint_pattern}"
        )

    ppl_evaluator = None
    if not args.skip_ppl:
        ppl_evaluator = ExternalPPLEvaluator(
            model_name=str(preset["eval_model"]),
            batch_size=int(preset.get("ppl_batch_size", 4)),
            max_length=int(preset.get("ppl_max_length", 512)),
            device=device,
        )

    qa_records = load_qa_prompt_records(args.qa_file)

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "task": args.task,
        "preset": args.preset,
        "device": str(device),
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_pattern": args.checkpoint_pattern,
        "num_samples": num_samples,
        "sampling": {
            "num_steps": sampling_cfg.num_steps,
            "sampling_method": sampling_cfg.sampling_method,
            "eta": sampling_cfg.eta,
            "temperature": sampling_cfg.temperature,
            "top_p": sampling_cfg.top_p,
            "top_k": sampling_cfg.top_k,
            "ban_special_tokens": sampling_cfg.ban_special_tokens,
            "adaptive_lm_guidance_scale": sampling_cfg.adaptive_lm_guidance_scale,
            "adaptive_lm_model_name": sampling_cfg.adaptive_lm_model_name,
            "adaptive_lm_guidance_mode": sampling_cfg.adaptive_lm_guidance_mode,
            "adaptive_lm_guidance_min_scale": sampling_cfg.adaptive_lm_guidance_min_scale,
            "adaptive_lm_guidance_scale_conditioned": sampling_cfg.adaptive_lm_guidance_scale_conditioned,
            "max_decode_tokens": sampling_cfg.max_decode_tokens,
        },
        "checkpoints": [],
    }

    print(f"Using device: {device}")
    print(f"Found {len(records)} continuous checkpoints")

    for idx, record in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] Processing {record.run_name}")

        checkpoint_result: Dict[str, Any] = {
            "run_name": record.run_name,
            "checkpoint_path": str(record.checkpoint_path),
            "metadata": dict(record.metadata),
            "tasks": {},
        }

        try:
            model, tokenizer, model_config = load_trainer_from_checkpoint(record.checkpoint_path, device)
            is_decoder_only = int(getattr(model_config.algo, "stage", -1)) == 2
            checkpoint_result["metadata"]["is_decoder_only"] = is_decoder_only
            checkpoint_result["metadata"]["effective_denoiser_parameterization"] = getattr(
                getattr(model, "denoiser", None), "parameterization", None
            )

            if is_decoder_only:
                finetune_path = getattr(getattr(model_config, "training", None), "finetune_path", None)
                checkpoint_result["metadata"]["training_finetune_path"] = finetune_path
                if finetune_path:
                    finetune_run_name = Path(str(finetune_path)).parent.parent.name
                    checkpoint_result["metadata"]["training_finetune_run_name"] = finetune_run_name

                    algo_param = str(checkpoint_result["metadata"].get("algo_parameterization"))
                    expected_prefix = {
                        "x0": "denoiser_x0",
                        "v": "denoiser_v",
                        "epsilon": "denoiser_epsilon",
                    }.get(algo_param)
                    checkpoint_result["metadata"]["pairing_expected_prefix"] = expected_prefix
                    checkpoint_result["metadata"]["pairing_ok"] = (
                        finetune_run_name.startswith(expected_prefix) if expected_prefix else None
                    )

            sources, source_metadata = _prepare_sources(
                model_config=model_config,
                tokenizer=tokenizer,
                fixed_texts_file=args.fixed_texts_file,
                num_samples=num_samples,
                device=device,
                seed=args.seed,
            )
            checkpoint_result["metadata"].update(source_metadata)

            if not sources and any(t in task_names for t in ("reconstruction", "infill")):
                raise RuntimeError("No evaluation sources available (validation and fixed texts unavailable)")

            for task_name in task_names:
                if _skip_stage2_task(task_name, is_decoder_only, args.force_stage2_tasks):
                    checkpoint_result["tasks"][task_name] = {
                        "status": "skipped",
                        "reason": "stage-2 decoder-only checkpoint (use --force_stage2_tasks to override)",
                    }
                    continue

                if task_name == "reconstruction":
                    checkpoint_result["tasks"][task_name] = run_reconstruction_encode_decode(
                        model,
                        tokenizer,
                        sources=sources,
                        batch_size=reconstruction_batch_size,
                        save_samples=args.save_samples,
                        sample_save_limit=sample_save_limit,
                    )
                elif task_name == "unconditional":
                    checkpoint_result["tasks"][task_name] = run_unconditional_generation(
                        model,
                        tokenizer,
                        num_samples=num_samples,
                        generation_batch_size=generation_batch_size,
                        sampling_cfg=sampling_cfg,
                        ppl_evaluator=ppl_evaluator,
                        save_samples=args.save_samples,
                        sample_save_limit=sample_save_limit,
                    )
                elif task_name == "infill":
                    checkpoint_result["tasks"][task_name] = run_infilling(
                        model,
                        tokenizer,
                        sources=sources,
                        num_samples=num_samples,
                        generation_batch_size=generation_batch_size,
                        sampling_cfg=sampling_cfg,
                        span_fraction=float(preset.get("infill_span_fraction", 0.25)),
                        min_span_tokens=int(preset.get("infill_min_span_tokens", 4)),
                        seed=args.seed,
                        save_samples=args.save_samples,
                        sample_save_limit=sample_save_limit,
                    )
                elif task_name == "prompt_qa":
                    checkpoint_result["tasks"][task_name] = run_prompt_qa(
                        model,
                        tokenizer,
                        qa_records=qa_records,
                        sampling_cfg=sampling_cfg,
                        default_max_answer_tokens=int(preset.get("qa_max_answer_tokens", 32)),
                        save_samples=args.save_samples,
                        sample_save_limit=sample_save_limit,
                    )
                else:
                    raise ValueError(f"Unsupported task: {task_name}")

        except Exception as exc:  # pylint: disable=broad-except
            checkpoint_result["error"] = str(exc)

        finally:
            if "model" in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        payload["checkpoints"].append(checkpoint_result)
        print(_record_summary_line(checkpoint_result))

    json_path = write_results_json(output_dir, payload)
    csv_path = write_results_csv(output_dir, payload)
    sample_paths = write_sample_artifacts(output_dir, payload) if args.save_samples else []

    payload["artifacts"] = {
        "results_json": str(json_path),
        "results_csv": str(csv_path),
        "sample_files": [str(p) for p in sample_paths],
    }

    # Update JSON with artifact references.
    write_results_json(output_dir, payload)

    print(f"Results JSON: {json_path}")
    print(f"Results CSV: {csv_path}")
    if sample_paths:
        print(f"Sample artifacts: {len(sample_paths)} files")

    return payload



def main() -> None:
    args = parse_args()
    payload = run_evaluation(args)
    # Human-readable summary for shell usage.
    print(json.dumps({"checkpoints": len(payload.get("checkpoints", []))}, indent=2))


if __name__ == "__main__":
    main()
