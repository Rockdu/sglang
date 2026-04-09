#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare no-rollout vs rollout-ode outputs across parallel configs."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Prefer repo python/ so local code changes are used.
_script_dir = Path(__file__).resolve().parent
_repo_python = _script_dir / "python"
if _repo_python.is_dir():
    sys.path.insert(0, str(_repo_python))
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(_repo_python) + (os.pathsep + _existing if _existing else "")

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from sglang.multimodal_gen import DiffGenerator


@dataclass(frozen=True)
class ParallelConfig:
    name: str
    tp_size: int
    sp_degree: int
    enable_cfg_parallel: bool


def parse_size(size: str) -> tuple[int, int]:
    width_str, height_str = size.strip().lower().split("x")
    return int(width_str), int(height_str)


def default_parallel_configs(num_gpus: int) -> list[ParallelConfig]:
    configs: list[ParallelConfig] = []
    for tp in range(1, num_gpus + 1):
        if num_gpus % tp == 0:
            sp = num_gpus // tp
            configs.append(
                ParallelConfig(
                    name=f"tp{tp}_sp{sp}_cfg0",
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
                        name=f"tp{tp}_sp{sp}_cfg1",
                        tp_size=tp,
                        sp_degree=sp,
                        enable_cfg_parallel=True,
                    )
                )
    dedup: dict[str, ParallelConfig] = {}
    for c in configs:
        dedup.setdefault(c.name, c)
    return list(dedup.values())


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


def run_one(
    generator: DiffGenerator,
    *,
    prompt: str,
    seed: int,
    size: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    negative_prompt: str | None,
    rollout: bool,
    noise_level: float,
    log_prob_no_const: bool,
) -> tuple[np.ndarray, str]:
    width, height = parse_size(size)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "return_file_paths_only": True,
        "rollout": rollout,
    }
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt

    if rollout:
        kwargs["rollout_sde_type"] = "ode"
        kwargs["rollout_noise_level"] = noise_level
        kwargs["rollout_log_prob_no_const"] = log_prob_no_const
        kwargs["rollout_debug_mode"] = False

    result = generator.generate(sampling_params_kwargs=kwargs)
    if isinstance(result, list):
        result = result[0] if result else None
    if result is None:
        raise RuntimeError("generator returned None")
    image_path = getattr(result, "output_file_path", None)
    if not image_path:
        raise RuntimeError("output_file_path missing in generation result")
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    return img, image_path


def cosine_similarity_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).reshape(-1)
    b64 = b.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0.0:
        return 1.0 if np.linalg.norm(a64) == 0.0 and np.linalg.norm(b64) == 0.0 else 0.0
    cos = float(np.dot(a64, b64) / denom)
    return float(np.clip(cos, -1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare no-rollout vs rollout-ode outputs across parallel configs."
    )
    parser.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--prompt", type=str, default="A cat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=str, default="1024x1024")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=3.0,
        help="Fallback guidance scale when cfg_parallel is enabled and guidance_scale is missing/<=1.",
    )
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--logprob-no-const", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="GPU count for matrix run. Default: auto-detect visible GPUs.",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sp-degree", type=int, default=1)
    parser.add_argument("--enable-cfg-parallel", action="store_true")
    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Only run one specified config (--tp-size/--sp-degree/--enable-cfg-parallel).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rollout_trajectory_debug_compare/ode_vs_no_rollout",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="ode_vs_no_rollout_report.md",
        help="Markdown report filename under run output directory.",
    )
    parser.add_argument(
        "--max-allowed-abs-diff",
        type=float,
        default=0.0,
        help="Max allowed per-pixel abs diff. 0 means exact match required.",
    )
    parser.add_argument(
        "--max-allowed-mse",
        type=float,
        default=0.0,
        help="Max allowed MSE. 0 means exact match required.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when any config exceeds thresholds or errors.",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    visible = (
        len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        if os.environ.get("CUDA_VISIBLE_DEVICES")
        else 1
    )
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        try:
            import torch

            visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 1
        except Exception:
            visible = 1
    effective_gpus = visible if args.num_gpus is None else min(args.num_gpus, visible)
    effective_gpus = max(effective_gpus, 1)

    if args.single_config:
        configs = [
            ParallelConfig(
                name=f"tp{args.tp_size}_sp{args.sp_degree}_cfg{int(args.enable_cfg_parallel)}",
                tp_size=args.tp_size,
                sp_degree=args.sp_degree,
                enable_cfg_parallel=args.enable_cfg_parallel,
            )
        ]
    else:
        configs = default_parallel_configs(effective_gpus)
        if not configs:
            configs = [
                ParallelConfig("tp1_sp1_cfg0", 1, 1, False),
            ]

    rows: list[dict[str, Any]] = []
    for cfg in configs:
        status = "ERROR"
        reason = ""
        base_path = "-"
        ode_path = "-"
        shape_str = "-"
        exact_match = False
        max_abs_diff: float | None = None
        mse: float | None = None
        cos: float | None = None

        try:
            generator = create_generator(
                model=args.model,
                num_gpus=effective_gpus,
                tp_size=cfg.tp_size,
                sp_degree=cfg.sp_degree,
                enable_cfg_parallel=cfg.enable_cfg_parallel,
                trust_remote_code=args.trust_remote_code,
                output_path=out_root,
            )
        except Exception as e:
            generator = None
            reason = f"Generator init failed: {e}"

        if generator is not None:
            try:
                gs = args.guidance_scale
                if cfg.enable_cfg_parallel and (gs is None or gs <= 1.0):
                    gs = args.cfg_guidance_scale
                neg_prompt = args.negative_prompt
                if gs is not None and gs > 1.0 and neg_prompt is None:
                    # Match trajectory debug test behavior for CFG-like runs.
                    neg_prompt = "low quality"
                base_img, base_path_real = run_one(
                    generator,
                    prompt=args.prompt,
                    seed=args.seed,
                    size=args.size,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=gs,
                    negative_prompt=neg_prompt,
                    rollout=False,
                    noise_level=args.noise_level,
                    log_prob_no_const=args.logprob_no_const,
                )
                ode_img, ode_path_real = run_one(
                    generator,
                    prompt=args.prompt,
                    seed=args.seed,
                    size=args.size,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=gs,
                    negative_prompt=neg_prompt,
                    rollout=True,
                    noise_level=args.noise_level,
                    log_prob_no_const=args.logprob_no_const,
                )
                base_path = base_path_real
                ode_path = ode_path_real

                if base_img.shape != ode_img.shape:
                    status = "FAIL"
                    reason = (
                        f"Image shapes mismatch: no_rollout={base_img.shape}, "
                        f"ode_rollout={ode_img.shape}"
                    )
                else:
                    shape_str = str(base_img.shape)
                    diff = np.abs(base_img.astype(np.int16) - ode_img.astype(np.int16))
                    max_abs_diff = float(diff.max()) if diff.size else 0.0
                    mse = float(
                        np.mean(
                            (base_img.astype(np.float64) - ode_img.astype(np.float64)) ** 2
                        )
                    )
                    cos = cosine_similarity_uint8(base_img, ode_img)
                    exact_match = bool(np.array_equal(base_img, ode_img))
                    within_threshold = (
                        max_abs_diff <= args.max_allowed_abs_diff
                        and mse <= args.max_allowed_mse
                    )
                    status = "PASS" if within_threshold else "FAIL"
                    if status == "FAIL":
                        reason = (
                            "Exceeded thresholds: "
                            f"max_abs_diff={max_abs_diff} (allowed {args.max_allowed_abs_diff}), "
                            f"mse={mse} (allowed {args.max_allowed_mse})"
                        )
            except Exception as e:
                status = "ERROR"
                reason = str(e)
            finally:
                generator.shutdown()

        rows.append(
            {
                "config": cfg.name,
                "tp_size": cfg.tp_size,
                "sp_degree": cfg.sp_degree,
                "enable_cfg_parallel": cfg.enable_cfg_parallel,
                "status": status,
                "reason": reason if reason else "-",
                "shape": shape_str,
                "exact_match": exact_match,
                "max_abs_diff": max_abs_diff,
                "mse": mse,
                "cosine_similarity": cos,
                "no_rollout_path": base_path,
                "rollout_ode_path": ode_path,
            }
        )

    overall_status = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"

    lines = [
        "# ODE vs No-Rollout Comparison Report",
        "",
        "## Result",
        "",
        f"- overall_status: **{overall_status}**",
        f"- tested_configs: `{len(rows)}`",
        "",
        "## Run Config",
        "",
        f"- model: `{args.model}`",
        f"- seed: `{args.seed}`",
        f"- size: `{args.size}`",
        f"- effective_gpus: `{effective_gpus}`",
        f"- single_config_mode: `{args.single_config}`",
        f"- guidance_scale: `{args.guidance_scale}`",
        f"- cfg_guidance_scale: `{args.cfg_guidance_scale}`",
        f"- num_inference_steps: `{args.num_inference_steps}`",
        f"- noise_level: `{args.noise_level}`",
        f"- logprob_no_const: `{args.logprob_no_const}`",
        "",
        "## Per-Config Summary",
        "",
        "| config | tp | sp | cfg_parallel | status | exact_match | max_abs_diff | mse | cosine_similarity | shape | reason |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['tp_size']} | {r['sp_degree']} | {r['enable_cfg_parallel']} | "
            f"{r['status']} | {r['exact_match']} | {r['max_abs_diff']} | {r['mse']} | "
            f"{r['cosine_similarity']} | {r['shape']} | {r['reason']} |"
        )

    lines.extend(
        [
            "",
            "## Output Paths",
            "",
            "| config | no_rollout | rollout_ode |",
            "|---|---|---|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['config']} | `{r['no_rollout_path']}` | `{r['rollout_ode_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            "| metric | value |",
            "|---|---|",
            f"| max_allowed_abs_diff | `{args.max_allowed_abs_diff}` |",
            f"| max_allowed_mse | `{args.max_allowed_mse}` |",
            "",
        ]
    )

    report_path = out_root / args.report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Markdown report written to {report_path}")
    print(f"overall_status={overall_status}")
    for r in rows:
        print(
            f"{r['config']}: status={r['status']}, exact_match={r['exact_match']}, "
            f"max_abs_diff={r['max_abs_diff']}, mse={r['mse']}"
        )

    if args.fail_on_mismatch and overall_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

