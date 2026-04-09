#!/usr/bin/env python3
"""Enumerate registered sglang-diffusion HF model paths, call ``POST /rollout/generate``, and write one Markdown report.

Tensor fields in JSON are replaced with ``__tensor_shape__`` / ``dtype`` (same idea as
``test_rollout_print_response_shapes.py``).

Model paths come from ``sglang.multimodal_gen.registry._MODEL_HF_PATH_TO_NAME`` (explicit
``hf_model_paths`` registrations). Detector-only families (e.g. GLM-Image, MOVA, LTX-2) are
not listed unless you pass ``--models``.

Usage:
    # Classify models and write a skeleton report (no GPU, no server)
    PYTHONPATH=python python test_rollout_all_models_response_shapes_report.py --dry-run

    # Full run: one server launch per model (needs weights on disk / Hugging Face cache)
    CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 \\
        PYTHONPATH=python python test_rollout_all_models_response_shapes_report.py \\
        --output rollout_response_shapes_report.md

    # Subset or connect to an already-running server for a single model
    ROLLOUT_TEST_BASE_URL=http://127.0.0.1:30000 \\
        PYTHONPATH=python python test_rollout_all_models_response_shapes_report.py \\
        --models Qwen/Qwen-Image --output qwen_shapes.md

    # Skip models that require an input image (I2V / I2I / I2M) unless you add
    # ``--fixture-image`` and ``--include-image-input-models``
    PYTHONPATH=python python test_rollout_all_models_response_shapes_report.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

# Repo ``python/`` layout (same as other rollout test scripts)
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_PYTHON = os.path.join(_REPO_ROOT, "python")
if _LOCAL_PYTHON not in sys.path:
    sys.path.insert(0, _LOCAL_PYTHON)

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType  # noqa: E402
from sglang.multimodal_gen.registry import (  # noqa: E402
    _MODEL_HF_PATH_TO_NAME,
    get_model_info,
)

PROMPT = "a red apple on a wooden table, simple"
SEED = 7
DEFAULT_PORT = int(os.environ.get("TEST_PORT", "39823"))


def tensors_to_shapes(obj: Any) -> Any:
    """Replace serialized tensor dicts with __tensor_shape__ + dtype; recurse."""
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            shape = obj.get("shape")
            dims = list(shape) if shape is not None else None
            out: dict[str, Any] = {"__tensor_shape__": dims}
            dt = obj.get("dtype")
            if dt is not None:
                out["dtype"] = dt
            return out
        return {k: tensors_to_shapes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensors_to_shapes(v) for v in obj]
    return obj


def wait_for_server(url: str, timeout: float = 900.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout}s: {url}")


def launch_server(
    *,
    port: int,
    num_gpus: int,
    tp_size: int | None,
    sp_degree: int | None,
    enable_cfg_parallel: bool,
    model_path: str,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    env["PYTHONPATH"] = _LOCAL_PYTHON + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.launch_server",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--num-gpus",
        str(num_gpus),
    ]
    if tp_size is not None:
        cmd.extend(["--tp-size", str(tp_size)])
    if sp_degree is not None:
        cmd.extend(["--sp-degree", str(sp_degree)])
    if enable_cfg_parallel:
        cmd.append("--enable-cfg-parallel")

    print("Launch:", " ".join(cmd), file=sys.stderr)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid,
    )


def kill_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        pass


def default_rollout_payload(
    *,
    fixture_image: list[str] | None,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "prompt": PROMPT,
        "seed": SEED,
        "num_inference_steps": 8,
        "guidance_scale": 4.0,
        "rollout_sde_type": "sde",
        "rollout_noise_level": 0.7,
        "rollout_debug_mode": True,
        "rollout_return_denoising_env": True,
        "rollout_return_dit_trajectory": True,
    }
    if fixture_image:
        p["image_path"] = fixture_image
    return p


def resolve_task_type(model_path: str) -> tuple[ModelTaskType | None, str | None]:
    """Return (task_type, error_message)."""
    try:
        info = get_model_info(model_path)
    except Exception as e:
        return None, f"get_model_info failed: {e}"
    if info is None:
        return None, "get_model_info returned None (unknown path / backend)"
    try:
        cfg = info.pipeline_config_cls()
        tt = getattr(cfg, "task_type", None)
        if isinstance(tt, ModelTaskType):
            return tt, None
        return None, f"pipeline_config has no ModelTaskType task_type: {type(tt)}"
    except Exception as e:
        return None, f"pipeline_config_cls() failed: {e}"


def discover_model_paths(extra: list[str] | None) -> list[str]:
    paths = sorted({*(_MODEL_HF_PATH_TO_NAME.keys()), *(extra or [])})
    return paths


@dataclass
class ModelRunResult:
    model_path: str
    task_type: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    http_status: int | None = None
    error_detail: str | None = None
    outline_json: str | None = None
    launch_cmd: str | None = None


def _md_escape_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    results: list[ModelRunResult],
    *,
    dry_run: bool,
    include_image_input: bool,
    fixture_image: list[str] | None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Rollout `/rollout/generate` response shape report",
        "",
        f"**Generated:** {ts}",
        "",
    ]
    if dry_run:
        lines.extend(
            [
                "> **Note:** This run used `--dry-run`: task types and the model list are filled in, but there are **no** per-model JSON shape outlines yet. On a machine with weights and GPU, run without `--dry-run` to record real `__tensor_shape__` trees.",
                "",
            ]
        )
    lines.extend(
        [
            "Serialized tensors in API JSON use `__tensor__` with base64 `data`. This report uses the same shape outline convention as `test_rollout_print_response_shapes.py`: each tensor becomes an object with `__tensor_shape__` and optional `dtype`.",
            "",
            "## Source of model list",
            "",
            "Paths are the union of `sglang.multimodal_gen.registry._MODEL_HF_PATH_TO_NAME` keys and any `--models` you pass. Entries that only use `model_detectors` (no explicit HF id in the registry) are **not** auto-included.",
            "",
            "## Run configuration",
            "",
            f"- **Dry run:** `{dry_run}` (no HTTP, no server).",
            f"- **Include image-input models (I2V / I2I / I2M):** `{include_image_input}`.",
            f"- **Fixture image paths:** `{fixture_image!r}`.",
            "",
            "## Summary",
            "",
            "| Model | Task type | Status | Notes |",
            "|-------|-----------|--------|-------|",
        ]
    )

    for r in results:
        st = "SKIPPED" if r.skipped else ("OK" if r.outline_json else "FAILED")
        notes = r.skip_reason or r.error_detail or ""
        lines.append(
            f"| `{r.model_path}` | {r.task_type or '—'} | {st} | {_md_escape_cell(notes[:200])} |"
        )

    lines.extend(["", "## Per-model details", ""])

    for r in results:
        lines.append(f"### `{r.model_path}`")
        lines.append("")
        lines.append(f"- **Task type:** {r.task_type or 'unknown'}")
        if r.launch_cmd:
            lines.append(f"- **Launch:** `{r.launch_cmd}`")
        if r.skipped:
            lines.append(f"- **Skipped:** {r.skip_reason}")
        elif r.outline_json:
            lines.append("- **Status:** OK")
            lines.append("")
            lines.append("```json")
            lines.append(r.outline_json)
            lines.append("```")
        else:
            lines.append(f"- **Status:** FAILED (HTTP {r.http_status})")
            if r.error_detail:
                lines.append("")
                lines.append("```text")
                lines.append(r.error_detail[:8000])
                lines.append("```")
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run rollout shape outline for registered diffusion models; write Markdown."
    )
    p.add_argument(
        "--output",
        "-o",
        default="rollout_response_shapes_report.md",
        help="Markdown output path (default: rollout_response_shapes_report.md).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="If set, use only these model paths (do not merge with the full registry list).",
    )
    p.add_argument(
        "--extra-models",
        nargs="*",
        default=[],
        help="Additional model paths to append to the registry list.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve task types and write the report skeleton; no server or HTTP.",
    )
    p.add_argument(
        "--include-image-input-models",
        action="store_true",
        help="Try I2V/I2I/I2M models (requires --fixture-image with real files).",
    )
    p.add_argument(
        "--fixture-image",
        nargs="*",
        default=None,
        help="Optional image_path list forwarded to the rollout request (for I2I/I2V).",
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--num-gpus", type=int, default=int(os.environ.get("TEST_NUM_GPUS", "1")))
    p.add_argument("--tp-size", type=int, default=None)
    p.add_argument("--sp-degree", type=int, default=None)
    p.add_argument(
        "--enable-cfg-parallel",
        "--cfgp",
        action="store_true",
    )
    p.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Stop after this many non-skipped attempts (useful for smoke tests).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.models:
        paths = list(args.models)
    else:
        paths = discover_model_paths(args.extra_models)

    skip_launch = os.environ.get("ROLLOUT_TEST_BASE_URL")
    if skip_launch and len(paths) > 1:
        print(
            "ROLLOUT_TEST_BASE_URL is set but multiple models were selected; "
            "the server loads one checkpoint — only the first path will be exercised.",
            file=sys.stderr,
        )
        paths = paths[:1]

    base_url = skip_launch.rstrip("/") if skip_launch else f"http://127.0.0.1:{args.port}"

    results: list[ModelRunResult] = []
    attempted = 0

    for model_path in paths:
        tt, err = resolve_task_type(model_path)
        tt_name = tt.name if tt else None
        res = ModelRunResult(model_path=model_path, task_type=tt_name)

        needs_image = tt.requires_image_input() if tt else False
        if needs_image and not args.include_image_input_models:
            res.skipped = True
            res.skip_reason = (
                "Model task requires image input (I2V/I2I/I2M); "
                "re-run with --include-image-input-models and --fixture-image."
            )
            results.append(res)
            continue

        if needs_image and args.include_image_input_models and not args.fixture_image:
            res.skipped = True
            res.skip_reason = (
                "--include-image-input-models set but no --fixture-image provided."
            )
            results.append(res)
            continue

        if err and tt is None:
            res.skipped = True
            res.skip_reason = err
            results.append(res)
            continue

        payload = default_rollout_payload(fixture_image=args.fixture_image)

        if args.dry_run:
            res.skipped = True
            res.skip_reason = "dry-run (no server started, no HTTP request)"
            results.append(res)
            continue

        if args.max_models is not None and attempted >= args.max_models:
            res.skipped = True
            res.skip_reason = f"stopped: --max-models {args.max_models} already reached"
            results.append(res)
            continue

        proc: subprocess.Popen | None = None
        attempted += 1
        try:
            if not skip_launch:
                proc = launch_server(
                    port=args.port,
                    num_gpus=args.num_gpus,
                    tp_size=args.tp_size,
                    sp_degree=args.sp_degree,
                    enable_cfg_parallel=args.enable_cfg_parallel,
                    model_path=model_path,
                )
                wait_for_server(base_url)

            res.launch_cmd = (
                f"ROLLOUT_TEST_BASE_URL={base_url}  (external server)"
                if skip_launch
                else (
                    f"python -m sglang.multimodal_gen.runtime.launch_server "
                    f"--model-path {model_path} --port {args.port} --num-gpus {args.num_gpus}"
                )
            )

            r = httpx.post(f"{base_url}/rollout/generate", json=payload, timeout=1200.0)
            res.http_status = r.status_code
            if r.status_code != 200:
                res.error_detail = r.text
            else:
                body = r.json()
                outline = tensors_to_shapes(body)
                res.outline_json = json.dumps(outline, indent=2, ensure_ascii=False)
        except Exception as e:
            res.error_detail = f"{type(e).__name__}: {e}"
        finally:
            if not skip_launch:
                kill_server(proc)

        results.append(res)

    md = render_markdown(
        results,
        dry_run=args.dry_run,
        include_image_input=args.include_image_input_models,
        fixture_image=args.fixture_image,
    )
    out_path = os.path.abspath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {out_path}", file=sys.stderr)

    failed = [r for r in results if not r.skipped and not r.outline_json]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
