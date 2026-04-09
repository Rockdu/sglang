#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run local DiffGenerator-based rollout validation for SDE/CPS and emit a markdown report.

This script does NOT call HTTP endpoints. It launches/uses local scheduler workers via
`DiffGenerator.from_pretrained(local_mode=True, ...)`.

Test items:
1) For fixed prompt/seed, compare SDE/CPS (noise_level=0) image error against rollout disabled.
2) For fixed prompt/seed, compare SDE/CPS images across noise levels [0, 0.2, 0.4, 0.6, 0.8].
3) On 4-GPU parallel configurations (SP/CFGP/TP combos), verify trajectory_log_probs consistency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Local test utility: avoid hard failure on flashinfer wheel/cache version mismatch.
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import GenerationResult


@dataclass(frozen=True)
class ParallelConfig:
    name: str
    tp_size: int
    sp_degree: int
    enable_cfg_parallel: bool


@dataclass
class ImageCase:
    mode: str
    noise_level: float
    output_file_path: str
    sha256: str


@dataclass
class LogProbCase:
    config: ParallelConfig
    mode: str
    noise_level: float
    shape: tuple[int, ...]
    values: np.ndarray
    output_file_path: str | None = None  # 生成图像路径，用于算输出误差


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SDE/CPS rollout behavior with DiffGenerator and write markdown report."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Model path/id to test.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Fixed test prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed.")
    parser.add_argument(
        "--size",
        type=str,
        default="1024x1024",
        help="Image size, e.g. 1024x1024.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Optional inference step override.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Optional guidance scale override.",
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=3.0,
        help=(
            "Guidance scale used for cfg-parallel cases in test #3 when --guidance-scale is "
            "unset or <= 1.0. Must be > 1.0 to enable CFG path."
        ),
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0,0.2,0.4,0.6,0.8",
        help="Comma-separated rollout noise levels for test #2.",
    )
    parser.add_argument(
        "--single-gpu-count",
        type=int,
        default=1,
        help="GPU count for test #1/#2.",
    )
    parser.add_argument(
        "--parallel-gpu-count",
        type=int,
        default=None,
        help="GPU count for test #3 parallel combinations. Default: auto-detect from visible GPUs.",
    )
    parser.add_argument(
        "--logprob-noise-level",
        type=float,
        default=0.6,
        help="Noise level for test #3 logprob consistency.",
    )
    parser.add_argument(
        "--logprob-no-const",
        action="store_true",
        help="Use rollout_log_prob_no_const=True in test #3.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Report/output directory. Default: outputs/diffgenerator_rollout_<timestamp>",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through trust_remote_code=True to DiffGenerator.",
    )
    return parser.parse_args()


def parse_noise_levels(arg: str) -> list[float]:
    levels = [float(x.strip()) for x in arg.split(",") if x.strip()]
    if not levels:
        raise ValueError("--noise-levels cannot be empty")
    return levels


def detect_visible_gpu_count() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        devices = [x.strip() for x in cvd.split(",") if x.strip()]
        if devices:
            return len(devices)
    try:
        import torch

        cnt = int(torch.cuda.device_count())
        if cnt > 0:
            return cnt
    except Exception:
        pass
    return 1


def parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format: {size!r}. Expected e.g. 1024x1024")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size values: {size!r}. Width/height must be > 0")
    return width, height


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    return arr


def image_error_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))
    psnr = float("inf") if mse == 0 else float(20.0 * math.log10(255.0 / rmse))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs": max_abs,
        "psnr": psnr,
    }


def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        raise ValueError("trajectory_log_probs is None")
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def generate_once(
    generator: DiffGenerator,
    *,
    prompt: str,
    seed: int,
    size: str,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    rollout: bool,
    rollout_sde_type: str | None,
    rollout_noise_level: float | None,
    rollout_log_prob_no_const: bool,
    negative_prompt: str | None = None,
    output_file_name: str | None = None,
) -> GenerationResult:
    width, height = parse_size(size)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "rollout": rollout,
        "return_file_paths_only": True,
    }
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    if rollout and rollout_sde_type is not None:
        kwargs["rollout_sde_type"] = rollout_sde_type
    if rollout and rollout_noise_level is not None:
        kwargs["rollout_noise_level"] = rollout_noise_level
    if rollout:
        kwargs["rollout_log_prob_no_const"] = rollout_log_prob_no_const
    if output_file_name is not None:
        kwargs["output_file_name"] = output_file_name

    result = generator.generate(sampling_params_kwargs=kwargs)
    if result is None:
        raise RuntimeError("Generation returned None")
    if isinstance(result, list):
        if not result:
            raise RuntimeError("Generation returned empty list")
        return result[0]
    return result


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

    # Non-CFG: tp * sp = num_gpus
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

    # CFG: tp * sp * 2 = num_gpus
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

    # Deduplicate by name while preserving order
    dedup: dict[str, ParallelConfig] = {}
    for c in configs:
        dedup.setdefault(c.name, c)
    return list(dedup.values())


def run_test_1_and_2(args: argparse.Namespace, out_root: Path) -> tuple[dict[str, Any], dict[str, list[ImageCase]]]:
    img_out = out_root / "images"
    ensure_dir(img_out)

    generator = create_generator(
        model=args.model,
        num_gpus=args.single_gpu_count,
        tp_size=1,
        sp_degree=1,
        enable_cfg_parallel=False,
        trust_remote_code=args.trust_remote_code,
        output_path=img_out,
    )

    baseline_case: ImageCase | None = None
    by_mode: dict[str, list[ImageCase]] = {"sde": [], "cps": []}
    test1_metrics: dict[str, dict[str, float]] = {}

    try:
        baseline = generate_once(
            generator,
            prompt=args.prompt,
            seed=args.seed,
            size=args.size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            rollout=False,
            rollout_sde_type=None,
            rollout_noise_level=None,
            rollout_log_prob_no_const=args.logprob_no_const,
            output_file_name="baseline",
        )
        if baseline.output_file_path is None:
            raise RuntimeError("Baseline generation has no output_file_path")

        baseline_path = Path(baseline.output_file_path)
        baseline_case = ImageCase(
            mode="baseline",
            noise_level=-1.0,
            output_file_path=str(baseline_path),
            sha256=sha256_file(baseline_path),
        )
        baseline_img = load_image_rgb(baseline_path)

        levels = parse_noise_levels(args.noise_levels)
        if 0.0 not in levels:
            levels = [0.0] + levels

        for mode in ("sde", "cps"):
            for level in levels:
                # 与 report 表头 (mode, noise_level) 对应: 文件名为 {mode}_{noise_level}
                result = generate_once(
                    generator,
                    prompt=args.prompt,
                    seed=args.seed,
                    size=args.size,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    rollout=True,
                    rollout_sde_type=mode,
                    rollout_noise_level=level,
                    rollout_log_prob_no_const=args.logprob_no_const,
                    output_file_name=f"{mode}_{level}",
                )
                if result.output_file_path is None:
                    raise RuntimeError(f"{mode}@{level} generation has no output_file_path")
                case_path = Path(result.output_file_path)
                by_mode[mode].append(
                    ImageCase(
                        mode=mode,
                        noise_level=float(level),
                        output_file_path=str(case_path),
                        sha256=sha256_file(case_path),
                    )
                )

            zero_case = next(c for c in by_mode[mode] if abs(c.noise_level) < 1e-12)
            zero_img = load_image_rgb(Path(zero_case.output_file_path))
            test1_metrics[mode] = image_error_metrics(zero_img, baseline_img)
    finally:
        generator.shutdown()

    if baseline_case is None:
        raise RuntimeError("baseline case was not generated")

    return {
        "baseline": baseline_case,
        "zero_error": test1_metrics,
    }, by_mode


def run_test_3(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    logprob_out = out_root / "logprob"
    ensure_dir(logprob_out)

    modes = ("sde", "cps")
    requested = args.parallel_gpu_count
    visible = detect_visible_gpu_count()
    effective_parallel_gpus = visible if requested is None else min(requested, visible)
    if requested is not None and requested > visible:
        print(
            f"[warn] --parallel-gpu-count={requested} exceeds visible GPUs ({visible}); "
            f"using {effective_parallel_gpus} instead."
        )

    configs = default_parallel_configs(effective_parallel_gpus)
    all_cases: list[LogProbCase] = []
    failures: list[str] = []
    cfg_guidance_overrides: dict[str, float] = {}

    for cfg in configs:
        gen: DiffGenerator | None = None
        try:
            gen = create_generator(
                model=args.model,
                num_gpus=effective_parallel_gpus,
                tp_size=cfg.tp_size,
                sp_degree=cfg.sp_degree,
                enable_cfg_parallel=cfg.enable_cfg_parallel,
                trust_remote_code=args.trust_remote_code,
                output_path=logprob_out,
            )

            for mode in modes:
                guidance_scale_for_case = args.guidance_scale
                if cfg.enable_cfg_parallel and (
                    guidance_scale_for_case is None or guidance_scale_for_case <= 1.0
                ):
                    guidance_scale_for_case = args.cfg_guidance_scale
                    if guidance_scale_for_case <= 1.0:
                        raise ValueError(
                            "cfg-parallel test requires guidance_scale > 1.0; "
                            f"got cfg-guidance-scale={guidance_scale_for_case}"
                        )
                    cfg_guidance_overrides[cfg.name] = guidance_scale_for_case

                # 与 report 表头 (config, mode) 对应: 文件名为 {config}_{mode}
                result = generate_once(
                    gen,
                    prompt=args.prompt,
                    seed=args.seed,
                    size=args.size,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=guidance_scale_for_case,
                    rollout=True,
                    rollout_sde_type=mode,
                    rollout_noise_level=args.logprob_noise_level,
                    rollout_log_prob_no_const=args.logprob_no_const,
                    negative_prompt="low quality",
                    output_file_name=f"{cfg.name}_{mode}",
                )
                arr = to_numpy(result.trajectory_log_probs)
                all_cases.append(
                    LogProbCase(
                        config=cfg,
                        mode=mode,
                        noise_level=args.logprob_noise_level,
                        shape=tuple(int(x) for x in arr.shape),
                        values=arr,
                        output_file_path=getattr(result, "output_file_path", None),
                    )
                )
        except Exception as e:
            failures.append(f"{cfg.name}: {e}")
            traceback.print_exc()
        finally:
            if gen is not None:
                gen.shutdown()

    grouped: dict[str, list[LogProbCase]] = {"sde": [], "cps": []}
    for c in all_cases:
        grouped[c.mode].append(c)

    consistency: dict[str, list[dict[str, Any]]] = {"sde": [], "cps": []}
    for mode in modes:
        mode_cases = grouped[mode]
        if not mode_cases:
            continue
        ref = mode_cases[0]
        ref_img: np.ndarray | None = None
        if ref.output_file_path and Path(ref.output_file_path).exists():
            try:
                ref_img = load_image_rgb(Path(ref.output_file_path))
            except Exception:
                ref_img = None
        for c in mode_cases:
            same_shape = c.shape == ref.shape
            if same_shape:
                diff = np.abs(c.values - ref.values)
                max_abs_diff = float(np.max(diff)) if diff.size > 0 else 0.0
                mean_abs_diff = float(np.mean(diff)) if diff.size > 0 else 0.0
                allclose = bool(np.allclose(c.values, ref.values, rtol=1e-6, atol=1e-6))
            else:
                max_abs_diff = float("inf")
                mean_abs_diff = float("inf")
                allclose = False

            # 输出（图像）误差 vs reference
            output_error: dict[str, float] | None = None
            if ref_img is not None and c.output_file_path and Path(c.output_file_path).exists():
                try:
                    c_img = load_image_rgb(Path(c.output_file_path))
                    output_error = image_error_metrics(c_img, ref_img)
                except Exception:
                    output_error = None

            consistency[mode].append(
                {
                    "config": c.config.name,
                    "shape": list(c.shape),
                    "same_shape_as_ref": same_shape,
                    "allclose_to_ref": allclose,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                    "output_error": output_error,
                }
            )

    return {
        "cases": all_cases,
        "consistency": consistency,
        "failures": failures,
        "effective_parallel_gpus": effective_parallel_gpus,
        "cfg_guidance_overrides": cfg_guidance_overrides,
    }


def relpath_str(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def _render_run_info(args: argparse.Namespace) -> list[str]:
    lines: list[str] = []
    lines.append("# SGLang-D Rollout SDE/CPS Validation Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append("")
    lines.append(f"- Time: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Prompt: {args.prompt}")
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Size: {args.size}")
    lines.append(f"- Num inference steps: {args.num_inference_steps}")
    lines.append(f"- Guidance scale: {args.guidance_scale}")
    lines.append(f"- rollout_log_prob_no_const: {args.logprob_no_const}")
    lines.append("")
    return lines


def render_report_test12(
    args: argparse.Namespace,
    out_root: Path,
    test12_summary: dict[str, Any],
    test2_cases: dict[str, list[ImageCase]],
) -> str:
    """Render report sections 1 and 2 (noise_level=0 error + noise levels comparison)."""
    lines: list[str] = _render_run_info(args)

    baseline: ImageCase = test12_summary["baseline"]
    lines.append("## 1) noise_level=0 vs rollout-disabled image error")
    lines.append("")
    lines.append("- 图像命名与表格对应: baseline → `baseline.png`, SDE/CPS → `{mode}_{noise_level}.png` (如 sde_0.0.png)")
    lines.append("")
    lines.append(f"- Baseline image: `{relpath_str(baseline.output_file_path, out_root)}`")
    lines.append(f"- Baseline sha256: `{baseline.sha256}`")
    lines.append("")
    lines.append("| mode | MAE | MSE | RMSE | MaxAbs | PSNR(dB) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode in ("sde", "cps"):
        m = test12_summary["zero_error"][mode]
        lines.append(
            f"| {mode} | {m['mae']:.6f} | {m['mse']:.6f} | {m['rmse']:.6f} | {m['max_abs']:.6f} | {m['psnr']:.6f} |"
        )
    lines.append("")

    lines.append("## 2) noise levels output comparison (fixed prompt/seed)")
    lines.append("")
    lines.append("图像命名: `{mode}_{noise_level}.png`，与下表 (mode, noise_level) 一一对应。")
    lines.append("")
    for mode in ("sde", "cps"):
        mode_cases = sorted(test2_cases[mode], key=lambda x: x.noise_level)
        lines.append(f"### {mode.upper()}")
        lines.append("")
        lines.append("| noise_level | image_path | sha256 |")
        lines.append("|---:|---|---|")
        for c in mode_cases:
            lines.append(
                f"| {c.noise_level:.1f} | `{relpath_str(c.output_file_path, out_root)}` | `{c.sha256}` |"
            )

        zero_case = next(c for c in mode_cases if abs(c.noise_level) < 1e-12)
        zero_img = load_image_rgb(Path(zero_case.output_file_path))
        lines.append("")
        lines.append("Pairwise error to noise_level=0:")
        lines.append("")
        lines.append("| noise_level | MAE | MSE | RMSE | MaxAbs | PSNR(dB) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for c in mode_cases:
            img = load_image_rgb(Path(c.output_file_path))
            m = image_error_metrics(img, zero_img)
            lines.append(
                f"| {c.noise_level:.1f} | {m['mae']:.6f} | {m['mse']:.6f} | {m['rmse']:.6f} | {m['max_abs']:.6f} | {m['psnr']:.6f} |"
            )
        lines.append("")

    return "\n".join(lines)


def render_report_test3(
    args: argparse.Namespace,
    out_root: Path,
    test3: dict[str, Any],
) -> str:
    """Render section 3: logprob 误差 + 输出误差（均以 reference 为基准）."""
    lines: list[str] = []
    lines.append("## 3) Parallel combo: logprob 误差 + 输出误差")
    lines.append("")
    lines.append("以每种 mode 下第一个成功 config 为 reference，比较 **logprob 误差** 与 **输出（图像）误差**。")
    lines.append("")
    lines.append(
        f"- Tested noise_level: {args.logprob_noise_level}, modes: sde/cps, parallel_gpus: {test3.get('effective_parallel_gpus')}"
    )
    cfg_overrides = test3.get("cfg_guidance_overrides", {})
    if cfg_overrides:
        lines.append("- CFG guidance overrides applied:")
        for cfg_name, gs in cfg_overrides.items():
            lines.append(f"  - {cfg_name}: guidance_scale={gs}")
    lines.append("")
    failures = test3["failures"]
    if failures:
        lines.append("### Launch/Run Failures")
        lines.append("")
        for msg in failures:
            lines.append(f"- {msg}")
        lines.append("")

    for mode in ("sde", "cps"):
        # 表格行与图像文件名对应: config + mode -> {config}_{mode}.png
        image_suffix = f"_{mode}.png"
        lines.append(f"### {mode.upper()} (reference = first config)")
        lines.append("")
        lines.append("**logprob 误差** (trajectory_log_probs vs reference):")
        lines.append("")
        lines.append("| config | image | shape | same_shape_as_ref | allclose_to_ref | max_abs_diff | mean_abs_diff |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for row in test3["consistency"][mode]:
            max_abs = row["max_abs_diff"]
            mean_abs = row["mean_abs_diff"]
            max_abs_str = "inf" if math.isinf(max_abs) else f"{max_abs:.8f}"
            mean_abs_str = "inf" if math.isinf(mean_abs) else f"{mean_abs:.8f}"
            img_name = f"{row['config']}{image_suffix}"
            lines.append(
                "| {config} | `{img}` | {shape} | {same_shape_as_ref} | {allclose_to_ref} | {max_abs} | {mean_abs} |".format(
                    config=row["config"],
                    img=img_name,
                    shape=row["shape"],
                    same_shape_as_ref=row["same_shape_as_ref"],
                    allclose_to_ref=row["allclose_to_ref"],
                    max_abs=max_abs_str,
                    mean_abs=mean_abs_str,
                )
            )
        lines.append("")
        lines.append("**输出误差** (生成图像 vs reference):")
        lines.append("")
        lines.append("| config | image | MAE | MSE | RMSE | MaxAbs | PSNR(dB) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in test3["consistency"][mode]:
            img_name = f"{row['config']}{image_suffix}"
            oe = row.get("output_error")
            if oe is not None:
                lines.append(
                    "| {} | `{}` | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                        row["config"], img_name, oe["mae"], oe["mse"], oe["rmse"], oe["max_abs"], oe["psnr"]
                    )
                )
            else:
                lines.append(f"| {row['config']} | `{img_name}` | — | — | — | — | — |")
        lines.append("")

    return "\n".join(lines)


def render_report(
    args: argparse.Namespace,
    out_root: Path,
    test12_summary: dict[str, Any],
    test2_cases: dict[str, list[ImageCase]],
    test3: dict[str, Any],
) -> str:
    """Full report: test 1&2 + test 3."""
    if test12_summary is not None:
        part12 = render_report_test12(args, out_root, test12_summary, test2_cases)
    else:
        part12 = ""
    if test3 is not None:
        part3 = render_report_test3(args, out_root, test3)
    else:
        part3 = ""
    return "\n".join(part for part in (part12, part3) if part)


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / f"diffgenerator_rollout_{timestamp}"
    )
    ensure_dir(out_root)

    run_cfg = vars(args).copy()
    (out_root / "run_config.json").write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test12_summary, test2_cases = run_test_1_and_2(args, out_root)
    # test12_summary, test2_cases = None, None
    test3 = run_test_3(args, out_root)

    report = render_report(args, out_root, test12_summary, test2_cases, test3)
    report_path = out_root / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[done] report written to: {report_path}")


if __name__ == "__main__":
    main()
