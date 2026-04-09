#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare SDE/CPS/ODE rollout trajectory debug
(prev_sample_mean, noise_std_dev, variance_noise, model_output) across parallel configs.

All tensors are converted to float32 before numpy (bf16-safe). Reports per-step and overall
max absolute difference and allclose for each quantity.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Prefer repo python/ so rollout and scheduler fixes are used (client and server subprocess)
_script_dir = Path(__file__).resolve().parent
_repo_python = _script_dir / "python"
if _repo_python.is_dir():
    sys.path.insert(0, str(_repo_python))
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(_repo_python) + (os.pathsep + _existing if _existing else "")

import numpy as np

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from sglang.multimodal_gen import DiffGenerator


@dataclass(frozen=True)
class ParallelConfig:
    name: str
    tp_size: int
    sp_degree: int
    enable_cfg_parallel: bool


ROLLOUT_MODES: tuple[str, ...] = ("sde", "cps", "ode")


def parse_size(size: str) -> tuple[int, int]:
    w, h = size.strip().lower().split("x")
    return int(w), int(h)


def create_generator(
    *,
    model: str,
    num_gpus: int,
    tp_size: int,
    sp_degree: int,
    enable_cfg_parallel: bool,
    trust_remote_code: bool,
    output_path: Path,
) -> DiffGenerator:
    return DiffGenerator.from_pretrained(
        local_mode=True,
        model_path=model,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_degree=sp_degree,
        enable_cfg_parallel=enable_cfg_parallel,
        trust_remote_code=trust_remote_code,
        output_path=str(output_path),
    )


def default_parallel_configs(num_gpus: int) -> list[ParallelConfig]:
    configs: list[ParallelConfig] = []
    for tp in range(1, num_gpus + 1):
        if num_gpus % tp == 0:
            sp = num_gpus // tp
            configs.append(
                ParallelConfig(
                    f"tp{tp}_sp{sp}_cfg0",
                    tp_size=tp,
                    sp_degree=sp,
                    enable_cfg_parallel=False,
                )
            )
    if num_gpus % 2 == 0:
        half = num_gpus // 2
        for tp in range(1, half + 1):
            if half % tp == 0:
                sp = half // tp
                configs.append(
                    ParallelConfig(
                        f"tp{tp}_sp{sp}_cfg1",
                        tp_size=tp,
                        sp_degree=sp,
                        enable_cfg_parallel=True,
                    )
                )
    dedup: dict[str, ParallelConfig] = {}
    for c in configs:
        dedup.setdefault(c.name, c)
    return list(dedup.values())


def to_numpy_bf16_safe(x: Any) -> np.ndarray:
    """Convert tensor to numpy; use float32 to avoid bf16/fp16 unsupported by numpy."""
    if x is None:
        raise ValueError("tensor is None")
    if isinstance(x, np.ndarray):
        return x.astype(np.float32) if x.dtype == np.float16 else x
    if hasattr(x, "detach"):
        t = x.detach().cpu().float()
        return t.numpy()
    return np.asarray(x, dtype=np.float32)


def to_step_list(x: Any) -> list[Any] | None:
    """Normalize trajectory debug payload into a per-step list."""
    if x is None:
        return None
    if isinstance(x, list):
        return x

    # Tensor-like path: expected layout is [B, T, ...]
    if hasattr(x, "ndim") and int(x.ndim) >= 2:
        if hasattr(x, "unbind"):
            return list(x.unbind(dim=1))
        if isinstance(x, np.ndarray):
            return [x[:, i, ...] for i in range(x.shape[1])]

    # Fallback to single-step payload
    return [x]


def run_one(
    generator: DiffGenerator,
    *,
    prompt: str,
    seed: int,
    size: str,
    mode: str,
    noise_level: float,
    num_inference_steps: int | None,
    num_frames: int | None,
    guidance_scale: float | None,
    cfg_guidance_scale: float,
    log_prob_no_const: bool,
    negative_prompt: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    """Run one generation with rollout; return debug trajectories as numpy lists or None."""
    width, height = parse_size(size)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "rollout": True,
        "rollout_sde_type": mode,
        "rollout_noise_level": noise_level,
        "rollout_log_prob_no_const": log_prob_no_const,
        "rollout_debug_mode": True,
        "return_file_paths_only": True,
    }
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    if num_frames is not None:
        kwargs["num_frames"] = num_frames
    gs = guidance_scale if guidance_scale is not None and guidance_scale > 1.0 else cfg_guidance_scale
    kwargs["guidance_scale"] = gs
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt

    result = generator.generate(sampling_params_kwargs=kwargs)
    if result is None:
        return None
    if isinstance(result, list):
        result = result[0] if result else None
    if result is None:
        return None
    rollout_trajectory_data = getattr(result, "rollout_trajectory_data", None)
    debug_tensors = (
        getattr(rollout_trajectory_data, "rollout_debug_tensors", None)
        if rollout_trajectory_data is not None
        else None
    )
    v = getattr(debug_tensors, "rollout_variance_noises", None)
    p = getattr(debug_tensors, "rollout_prev_sample_means", None)
    s = getattr(debug_tensors, "rollout_noise_std_devs", None)
    m = getattr(debug_tensors, "rollout_model_outputs", None)
    lp = getattr(rollout_trajectory_data, "rollout_log_probs", None)
    v_steps = to_step_list(v)
    p_steps = to_step_list(p)
    s_steps = to_step_list(s)
    m_steps = to_step_list(m)
    lp_steps = to_step_list(lp)
    if not v_steps or not p_steps or not s_steps or not m_steps or not lp_steps:
        return None
    return (
        [to_numpy_bf16_safe(t) for t in v_steps],
        [to_numpy_bf16_safe(t) for t in p_steps],
        [to_numpy_bf16_safe(t) for t in s_steps],
        [to_numpy_bf16_safe(t) for t in m_steps],
        [to_numpy_bf16_safe(t) for t in lp_steps],
    )


def compare_lists(
    ref_v: list[np.ndarray],
    ref_p: list[np.ndarray],
    ref_s: list[np.ndarray],
    ref_m: list[np.ndarray],
    ref_lp: list[np.ndarray],
    cur_v: list[np.ndarray],
    cur_p: list[np.ndarray],
    cur_s: list[np.ndarray],
    cur_m: list[np.ndarray],
    cur_lp: list[np.ndarray],
) -> dict[str, Any]:
    """Compare current (cur_*) to reference (ref_*). Return metrics for each quantity."""

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a64 = a.astype(np.float64).reshape(-1)
        b64 = b.astype(np.float64).reshape(-1)
        denom = np.linalg.norm(a64) * np.linalg.norm(b64)
        if denom == 0.0:
            # Both zeros are identical directionally; one-sided zero should be worst-case.
            return 1.0 if np.linalg.norm(a64) == 0.0 and np.linalg.norm(b64) == 0.0 else 0.0
        cos = float(np.dot(a64, b64) / denom)
        # Guard against tiny floating-point overshoot.
        return float(np.clip(cos, -1.0, 1.0))

    def metrics(ref_list: list[np.ndarray], cur_list: list[np.ndarray], name: str) -> dict[str, Any]:
        def step_mean_abs_diff(r: np.ndarray, c: np.ndarray) -> float:
            return float(np.mean(np.abs(c.astype(np.float64) - r.astype(np.float64))))

        def step_max_abs_diff(r: np.ndarray, c: np.ndarray) -> float:
            return float(np.max(np.abs(c.astype(np.float64) - r.astype(np.float64))))

        bad = {
            f"{name}_same_len": False,
            f"{name}_same_shapes": False,
            f"{name}_max_abs_diff": float("inf"),
            f"{name}_first_step_mean_abs_diff": float("inf"),
            f"{name}_last_step_mean_abs_diff": float("inf"),
            f"{name}_first_step_cosine": -1.0,
            f"{name}_last_step_cosine": -1.0,
        }
        if len(cur_list) != len(ref_list):
            return bad
        if not all(c.shape == r.shape for c, r in zip(cur_list, ref_list)):
            bad[f"{name}_same_len"] = True
            return bad
        if len(ref_list) == 0:
            return {
                f"{name}_same_len": True,
                f"{name}_same_shapes": True,
                f"{name}_max_abs_diff": 0.0,
                f"{name}_first_step_mean_abs_diff": None,
                f"{name}_last_step_mean_abs_diff": None,
                f"{name}_first_step_cosine": None,
                f"{name}_last_step_cosine": None,
            }
        max_abs = max(step_max_abs_diff(ref_list[i], cur_list[i]) for i in range(len(ref_list)))
        return {
            f"{name}_same_len": True,
            f"{name}_same_shapes": True,
            f"{name}_max_abs_diff": max_abs,
            f"{name}_first_step_mean_abs_diff": step_mean_abs_diff(ref_list[0], cur_list[0]),
            f"{name}_last_step_mean_abs_diff": step_mean_abs_diff(ref_list[-1], cur_list[-1]),
            f"{name}_first_step_cosine": cosine_similarity(cur_list[0], ref_list[0]),
            f"{name}_last_step_cosine": cosine_similarity(cur_list[-1], ref_list[-1]),
        }

    out: dict[str, Any] = {}
    out.update(metrics(ref_v, cur_v, "variance_noise"))
    out.update(metrics(ref_p, cur_p, "prev_sample_mean"))
    out.update(metrics(ref_s, cur_s, "noise_std_dev"))
    out.update(metrics(ref_m, cur_m, "model_output"))
    out.update(metrics(ref_lp, cur_lp, "log_prob"))
    return out


def summarize_shapes(arr_list: list[np.ndarray]) -> str:
    """Summarize per-step tensor shapes for table output."""
    if not arr_list:
        return "-"
    shapes = [tuple(arr.shape) for arr in arr_list]
    unique = []
    for s in shapes:
        if s not in unique:
            unique.append(s)
    if len(unique) == 1:
        return f"steps={len(shapes)}, shape={unique[0]}"
    return "; ".join(f"s{i}:{s}" for i, s in enumerate(shapes))


def format_array_preview(arr: np.ndarray) -> str:
    """Return a truncated textual dump similar to tensor print preview."""
    return np.array2string(
        arr,
        threshold=32,
        edgeitems=2,
        max_line_width=140,
        separator=", ",
        precision=8,
        floatmode="maxprec_equal",
    )


def format_metric(v: Any, digits: int = 8) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if np.isfinite(v):
            return f"{float(v):.{digits}g}"
        return str(v)
    return str(v)


def write_tensor_dump_file(
    *,
    out_root: Path,
    data: dict[str, dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]]]],
    effective_gpus: int,
    args: argparse.Namespace,
    models: list[str],
    failures: list[str],
    modes: tuple[str, ...],
) -> Path:
    """Write all compared intermediate tensors in plain text."""
    lines: list[str] = [
        "# Rollout trajectory debug tensor dump",
        "",
        f"Effective GPUs: {effective_gpus}",
        f"Prompt: {args.prompt!r}, seed={args.seed}, noise_level={args.noise_level}",
        "",
    ]
    lines.append("Models:")
    for m in models:
        lines.append(f"- {m}")
    lines.append("")
    if failures:
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")

    for model in models:
        lines.append(f"## Model: {model}")
        lines.append("")
        for mode in modes:
            lines.append(f"### Mode: {mode}")
            lines.append("")
            mode_data = data.get(model, {}).get(mode, {})
            if not mode_data:
                lines.append("(no data)")
                lines.append("")
                continue

            for config_name, (v_steps, p_steps, s_steps, m_steps, lp_steps) in mode_data.items():
                lines.append(f"#### Config: {config_name}")
                lines.append("")
                max_steps = max(len(v_steps), len(p_steps), len(s_steps), len(m_steps), len(lp_steps))
                for step_idx in range(max_steps):
                    lines.append(f"Step {step_idx}:")
                    lines.append("")

                    if step_idx < len(v_steps):
                        v = v_steps[step_idx]
                        lines.append(f"- variance_noise shape={v.shape}")
                        lines.append(format_array_preview(v))
                        lines.append("")
                    else:
                        lines.append("- variance_noise: <missing>")
                        lines.append("")

                    if step_idx < len(p_steps):
                        p = p_steps[step_idx]
                        lines.append(f"- prev_sample_mean shape={p.shape}")
                        lines.append(format_array_preview(p))
                        lines.append("")
                    else:
                        lines.append("- prev_sample_mean: <missing>")
                        lines.append("")

                    if step_idx < len(s_steps):
                        s = s_steps[step_idx]
                        lines.append(f"- noise_std_dev shape={s.shape}")
                        lines.append(format_array_preview(s))
                        lines.append("")
                    else:
                        lines.append("- noise_std_dev: <missing>")
                        lines.append("")

                    if step_idx < len(m_steps):
                        m = m_steps[step_idx]
                        lines.append(f"- model_output shape={m.shape}")
                        lines.append(format_array_preview(m))
                        lines.append("")
                    else:
                        lines.append("- model_output: <missing>")
                        lines.append("")

                    if step_idx < len(lp_steps):
                        lp = lp_steps[step_idx]
                        lines.append(f"- log_prob shape={lp.shape}")
                        lines.append(format_array_preview(lp))
                        lines.append("")
                    else:
                        lines.append("- log_prob: <missing>")
                        lines.append("")
                    lines.append("")
        lines.append("")

    dump_path = out_root / "trajectory_debug_tensors.txt"
    dump_path.write_text("\n".join(lines), encoding="utf-8")
    return dump_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SDE/CPS/ODE trajectory debug (prev_sample_mean, noise_std_dev, variance_noise, model_output) across parallel configs."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "Qwen/Qwen-Image",
            "Tongyi-MAI/Z-Image-Turbo",
            "black-forest-labs/FLUX.1-dev",
        ],
        help="Model paths to test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single-model override (backward-compatible).",
    )
    parser.add_argument("--prompt", type=str, default="A cat", help="Prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--size", type=str, default="1024x1024", help="Image/video spatial size.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames for T2V models (None=use model default).")
    parser.add_argument("--noise-level", type=float, default=0.5, help="Rollout noise level.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--cfg-guidance-scale", type=float, default=3.0)
    parser.add_argument(
        "--logprob-no-const",
        action="store_true",
        help="Accepted for backward compatibility; this harness always passes rollout_log_prob_no_const=True.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--parallel-gpu-count", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=list(ROLLOUT_MODES),
        choices=list(ROLLOUT_MODES),
        metavar="MODE",
        help=f"Rollout SDE types to run/compare (default: all). Choices: {', '.join(ROLLOUT_MODES)}.",
    )
    args = parser.parse_args()

    active_modes: tuple[str, ...] = tuple(dict.fromkeys(args.modes))

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = Path("outputs/rollout_trajectory_debug_compare") / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_root}")

    model_list = [args.model] if args.model else list(args.models)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    model_list = [m for m in model_list if not (m in seen or seen.add(m))]
    print("Models:")
    for m in model_list:
        print(f"  - {m}")

    visible = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        try:
            import torch
            visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 1
        except Exception:
            visible = 1
    effective_gpus = visible if args.parallel_gpu_count is None else min(args.parallel_gpu_count, visible)
    configs = default_parallel_configs(effective_gpus)
    ref_config = ParallelConfig(
        name="single_gpu_ref",
        tp_size=1,
        sp_degree=1,
        enable_cfg_parallel=False,
    )

    # (model -> mode -> config_name -> (v_list, p_list, s_list, m_list))
    data: dict[str, dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]]]] = {}
    for model in model_list:
        data[model] = {mode: {} for mode in active_modes}
    failures: list[str] = []

    for model in model_list:
        # 1) Always run single-GPU reference first.
        gen_ref: DiffGenerator | None = None
        try:
            gen_ref = create_generator(
                model=model,
                num_gpus=1,
                tp_size=1,
                sp_degree=1,
                enable_cfg_parallel=False,
                trust_remote_code=args.trust_remote_code,
                output_path=out_root,
            )
            for mode in active_modes:
                tri = run_one(
                    gen_ref,
                    prompt=args.prompt,
                    seed=args.seed,
                    size=args.size,
                    mode=mode,
                    noise_level=args.noise_level,
                    num_inference_steps=args.num_inference_steps,
                    num_frames=args.num_frames,
                    guidance_scale=args.guidance_scale,
                    cfg_guidance_scale=args.cfg_guidance_scale,
                    log_prob_no_const=True,
                    negative_prompt="low quality",
                )
                if tri is not None:
                    data[model][mode][ref_config.name] = tri
                else:
                    failures.append(f"{model}:{ref_config.name}_{mode}: no trajectory debug data")
        except Exception as e:
            failures.append(f"{model}:{ref_config.name}: {e}")
            traceback.print_exc()
        finally:
            if gen_ref is not None:
                gen_ref.shutdown()

        # 2) Run parallel configs and compare against single-GPU reference.
        for cfg in configs:
            gen: DiffGenerator | None = None
            try:
                gen = create_generator(
                    model=model,
                    num_gpus=effective_gpus,
                    tp_size=cfg.tp_size,
                    sp_degree=cfg.sp_degree,
                    enable_cfg_parallel=cfg.enable_cfg_parallel,
                    trust_remote_code=args.trust_remote_code,
                    output_path=out_root,
                )
                gs = args.guidance_scale
                if cfg.enable_cfg_parallel and (gs is None or gs <= 1.0):
                    gs = args.cfg_guidance_scale
                for mode in active_modes:
                    tri = run_one(
                        gen,
                        prompt=args.prompt,
                        seed=args.seed,
                        size=args.size,
                        mode=mode,
                        noise_level=args.noise_level,
                        num_inference_steps=args.num_inference_steps,
                        num_frames=args.num_frames,
                        guidance_scale=gs,
                        cfg_guidance_scale=args.cfg_guidance_scale,
                        log_prob_no_const=True,
                        negative_prompt="low quality",
                    )
                    if tri is not None:
                        data[model][mode][cfg.name] = tri
                    else:
                        failures.append(f"{model}:{cfg.name}_{mode}: no trajectory debug data")
            except Exception as e:
                failures.append(f"{model}:{cfg.name}: {e}")
                traceback.print_exc()
            finally:
                if gen is not None:
                    gen.shutdown()

    # Build report: for each model/mode, all configs compare to single-GPU ref.
    report: dict[str, dict[str, list[dict[str, Any]]]] = {
        m: {mode: [] for mode in active_modes} for m in model_list
    }
    for model in model_list:
        for mode in active_modes:
            mode_data = data[model][mode]
            if ref_config.name not in mode_data:
                failures.append(f"{model}:{mode}: missing single-GPU reference")
                continue
            ref_v, ref_p, ref_s, ref_m, ref_lp = mode_data[ref_config.name]
            for cname in mode_data.keys():
                cur_v, cur_p, cur_s, cur_m, cur_lp = mode_data[cname]
                row = {"config": cname, "reference": ref_config.name}
                row.update(
                    compare_lists(
                        ref_v, ref_p, ref_s, ref_m, ref_lp,
                        cur_v, cur_p, cur_s, cur_m, cur_lp,
                    )
                )
                report[model][mode].append(row)

    # Print and write report
    print("=== Rollout trajectory debug (SDE/CPS/ODE; prev_sample_mean, noise_std_dev, variance_noise, model_output) ===\n")
    print(f"Effective GPUs: {effective_gpus}")
    print(f"Reference config for all comparisons: {ref_config.name}")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print()
    lines = [
        "# Rollout trajectory debug comparison",
        "",
        f"Effective GPUs: {effective_gpus}",
        f"Reference config for all comparisons: {ref_config.name}",
        f"Prompt: {args.prompt!r}, seed={args.seed}, noise_level={args.noise_level}",
        "",
    ]
    lines.append("Models:")
    for m in model_list:
        lines.append(f"- {m}")
    lines.append("")
    if failures:
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")
    for model in model_list:
        lines.append(f"## Model: {model}")
        lines.append("")
        for mode in active_modes:
            if not report[model][mode]:
                continue
            lines.append(f"### Mode: {mode}")
            lines.append("")
            for quantity, key, arr_index in (
                ("variance_noise", "variance_noise", 0),
                ("prev_sample_mean", "prev_sample_mean", 1),
                ("noise_std_dev", "noise_std_dev", 2),
                ("model_output", "model_output", 3),
                ("log_prob", "log_prob", 4),
            ):
                lines.append(f"#### {quantity}")
                lines.append("")
                lines.append("| config | reference | shape | max_abs_diff | first_step_mean_abs_diff | last_step_mean_abs_diff | first_step_cosine | last_step_cosine |")
                lines.append("|--------|-----------|-------|--------------|--------------------------|-------------------------|-------------------|------------------|")
                for row in report[model][mode]:
                    cur_list = data[model][mode][row["config"]][arr_index]
                    shape = summarize_shapes(cur_list)
                    max_abs = format_metric(row.get(f"{key}_max_abs_diff", ""))
                    first_mad = format_metric(row.get(f"{key}_first_step_mean_abs_diff", ""))
                    last_mad = format_metric(row.get(f"{key}_last_step_mean_abs_diff", ""))
                    first_cos = format_metric(row.get(f"{key}_first_step_cosine", ""))
                    last_cos = format_metric(row.get(f"{key}_last_step_cosine", ""))
                    lines.append(
                        f"| {row['config']} | {row['reference']} | {shape} | {max_abs} | {first_mad} | {last_mad} | {first_cos} | {last_cos} |"
                    )
                lines.append("")
    report_path = out_root / "trajectory_debug_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    dump_path = write_tensor_dump_file(
        out_root=out_root,
        data=data,
        effective_gpus=effective_gpus,
        args=args,
        models=model_list,
        failures=failures,
        modes=active_modes,
    )
    print(f"Tensor dump written to {dump_path}")


if __name__ == "__main__":
    main()
