#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decoupled check: multi-image vs repeated single-image generation under the seed contract.

``InputValidationStage._generate_seeds`` (see ``input_validation.py``) expands one base
``seed`` into ``[seed + i for i in range(num_outputs_per_prompt)]`` and builds one
``torch.Generator`` per output.

This script verifies that:

- One call with ``num_outputs_per_prompt=K`` and ``seed=S`` produces K images, and
- K separate calls with ``num_outputs_per_prompt=1`` and ``seed=S, S+1, ..., S+K-1``

produce **pairwise identical** RGB uint8 outputs (same prompt and other hyperparameters).

It does not replace parallel / rollout-ODE comparison tests; it only pins the
single-vs-multi batching contract.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_script_dir = Path(__file__).resolve().parent
_repo_python = _script_dir / "python"
if _repo_python.is_dir():
    sys.path.insert(0, str(_repo_python))
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(_repo_python) + (os.pathsep + _existing if _existing else "")

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from sglang.multimodal_gen import DiffGenerator


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


def _normalize_results(result: Any) -> list[Any]:
    if result is None:
        return []
    if not isinstance(result, list):
        return [result]
    return list(result)


def generate_images(
    generator: DiffGenerator,
    *,
    prompt: str,
    base_seed: int,
    num_outputs_per_prompt: int,
    width: int,
    height: int,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    negative_prompt: str | None,
    rollout: bool,
    noise_level: float,
    log_prob_no_const: bool,
) -> list[np.ndarray]:
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "seed": base_seed,
        "width": width,
        "height": height,
        "return_file_paths_only": True,
        "rollout": rollout,
        "num_outputs_per_prompt": num_outputs_per_prompt,
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

    raw = generator.generate(sampling_params_kwargs=kwargs)
    items = _normalize_results(raw)
    if len(items) != num_outputs_per_prompt:
        raise RuntimeError(
            f"expected {num_outputs_per_prompt} GenerationResult entries, got {len(items)}"
        )
    out: list[np.ndarray] = []
    for item in items:
        path = getattr(item, "output_file_path", None)
        if not path:
            raise RuntimeError("output_file_path missing on GenerationResult")
        out.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    return out


def compare_images(a: np.ndarray, b: np.ndarray) -> tuple[bool, float, float]:
    """Return (exact_match, max_abs_diff, mse)."""
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    max_abs = float(diff.max()) if diff.size else 0.0
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    exact = bool(np.array_equal(a, b))
    return exact, max_abs, mse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify K single generations (seeds S..S+K-1) match one multi generation "
            "(seed S, num_outputs_per_prompt=K)."
        )
    )
    parser.add_argument("--model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument("--prompt", type=str, default="A cat")
    parser.add_argument("--seed", type=int, default=42, help="Base seed S for the multi call.")
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=2,
        help="K: number of images (>=1). Singles use seeds S, S+1, ..., S+K-1.",
    )
    parser.add_argument("--size", type=str, default="1024x1024")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--logprob-no-const", action="store_true")
    parser.add_argument("--rollout", action="store_true", help="Use rollout ODE path for both runs.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sp-degree", type=int, default=1)
    parser.add_argument("--enable-cfg-parallel", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/seed_single_vs_multi",
        help="Directory root; a timestamped subdir is created per run.",
    )
    parser.add_argument(
        "--max-allowed-abs-diff",
        type=float,
        default=0.0,
        help="Per-pixel tolerance; 0 requires exact uint8 match.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with code 1 if any pair exceeds tolerance.",
    )
    args = parser.parse_args()

    if args.num_outputs < 1:
        raise SystemExit("--num-outputs must be >= 1")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    width, height = parse_size(args.size)
    gen = create_generator(
        model=args.model,
        num_gpus=args.num_gpus,
        tp_size=args.tp_size,
        sp_degree=args.sp_degree,
        enable_cfg_parallel=args.enable_cfg_parallel,
        trust_remote_code=args.trust_remote_code,
        output_path=out_root,
    )

    try:
        multi = generate_images(
            gen,
            prompt=args.prompt,
            base_seed=args.seed,
            num_outputs_per_prompt=args.num_outputs,
            width=width,
            height=height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            rollout=args.rollout,
            noise_level=args.noise_level,
            log_prob_no_const=args.logprob_no_const,
        )

        singles: list[np.ndarray] = []
        for i in range(args.num_outputs):
            one = generate_images(
                gen,
                prompt=args.prompt,
                base_seed=args.seed + i,
                num_outputs_per_prompt=1,
                width=width,
                height=height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                rollout=args.rollout,
                noise_level=args.noise_level,
                log_prob_no_const=args.logprob_no_const,
            )
            singles.append(one[0])

        lines = [
            "# Seed single vs multi generation",
            "",
            "Compare one request with `num_outputs_per_prompt=K` and base seed `S` against",
            "`K` requests with `num_outputs_per_prompt=1` and seeds `S..S+K-1` (see `InputValidationStage._generate_seeds`).",
            "",
            "## Config",
            "",
            f"- model: `{args.model}`",
            f"- prompt: `{args.prompt!r}`",
            f"- base_seed S: `{args.seed}`",
            f"- K: `{args.num_outputs}`",
            f"- size: `{args.size}`",
            f"- rollout_ode: `{args.rollout}`",
            f"- max_allowed_abs_diff: `{args.max_allowed_abs_diff}`",
            "",
            "## Per-index comparison",
            "",
            "| i | seed (single path) | exact_match | max_abs_diff | mse |",
            "|---|---:|---|---:|---:|",
        ]

        all_ok = True
        for i in range(args.num_outputs):
            ex, mad, mse = compare_images(multi[i], singles[i])
            ok = mad <= args.max_allowed_abs_diff
            all_ok = all_ok and ok
            lines.append(
                f"| {i} | {args.seed + i} | {ex} | {mad} | {mse:.6g} |"
            )
            print(
                f"[{i}] seed={args.seed + i} exact={ex} max_abs_diff={mad} mse={mse:.6g} ok={ok}"
            )

        lines.append("")
        lines.append(f"- **overall_pass**: `{all_ok}`")
        lines.append("")

        report_path = out_root / "seed_single_vs_multi_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report written to {report_path}")

        if args.fail_on_mismatch and not all_ok:
            raise SystemExit(1)
    finally:
        gen.shutdown()


if __name__ == "__main__":
    main()
