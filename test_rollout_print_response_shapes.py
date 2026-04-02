#!/usr/bin/env python3
"""Start sglang-D with Qwen-Image, call rollout with return_dit_env and rollout_debug_mode, print shape outline.

POST /rollout/images returns JSON where tensors are dicts:
  {"__tensor__": true, "data": "<base64>", "shape": [...], "dtype": "..."}

This script prints the same structure as a dict/JSON but replaces every tensor with
a tagged object: dict with key __tensor_shape__ (and dtype when present), so tensor
shapes are not confused with ordinary JSON lists left as-is by the server.

Usage:
    CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 \\
        python test_rollout_print_response_shapes.py

    # Example: 4 GPUs, TP=2, SP=2, CFG parallel (matches sglang-D CLI flags)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python test_rollout_print_response_shapes.py \\
        --num-gpus 4 --tp-size 2 --sp-degree 2 --enable-cfg-parallel

If a server is already running, set ROLLOUT_TEST_BASE_URL (e.g. http://127.0.0.1:30000)
to skip launching a child process (TP/SP/CFGP flags are then ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

import httpx

MODEL = "Qwen/Qwen-Image"
DEFAULT_PORT = int(os.environ.get("TEST_PORT", "39822"))

PROMPT = "a red apple on a wooden table, simple"
SEED = 7


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


def wait_for_server(url: str, timeout: float = 600.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                print(f"Server ready in {time.time() - start:.1f}s", file=sys.stderr)
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
    repo_root = os.path.dirname(os.path.abspath(__file__))
    local_python = os.path.join(repo_root, "python")
    env["PYTHONPATH"] = local_python + os.pathsep + env.get("PYTHONPATH", "")

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print rollout /rollout/images response with tensor shapes tagged."
    )
    p.add_argument(
        "--model-path",
        default=os.environ.get("ROLLOUT_TEST_MODEL", MODEL),
        help="HuggingFace model id or path (default: env ROLLOUT_TEST_MODEL or Qwen/Qwen-Image).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"HTTP port for launch_server (default: env TEST_PORT or {DEFAULT_PORT}).",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("TEST_NUM_GPUS", "1")),
        help="launch_server --num-gpus (use ≥2 when using TP/SP/CFG parallel).",
    )
    p.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size (launch_server --tp-size). Omit for default 1.",
    )
    p.add_argument(
        "--sp-degree",
        type=int,
        default=None,
        help="Sequence parallel degree (launch_server --sp-degree). Omit for server default/auto.",
    )
    p.add_argument(
        "--enable-cfg-parallel",
        "--cfgp",
        action="store_true",
        help="Enable CFG parallel (launch_server --enable-cfg-parallel).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    proc: subprocess.Popen | None = None
    skip_launch = os.environ.get("ROLLOUT_TEST_BASE_URL")
    base = skip_launch if skip_launch else f"http://127.0.0.1:{args.port}"
    base = base.rstrip("/")

    try:
        if not skip_launch:
            proc = launch_server(
                port=args.port,
                num_gpus=args.num_gpus,
                tp_size=args.tp_size,
                sp_degree=args.sp_degree,
                enable_cfg_parallel=args.enable_cfg_parallel,
                model_path=args.model_path,
            )
            wait_for_server(base)

        payload = {
            "prompt": PROMPT,
            "seed": SEED,
            "num_inference_steps": 8,
            "guidance_scale": 4.0,
            "rollout_sde_type": "sde",
            "rollout_noise_level": 0.7,
            "rollout_debug_mode": True,
            "return_trajectory_latents": False,
            "return_trajectory_decoded": False,
            "return_dit_env": True,
        }

        print("POST /rollout/images", json.dumps(payload, indent=2), sep="\n", file=sys.stderr)
        r = httpx.post(f"{base}/rollout/images", json=payload, timeout=600)
        if r.status_code != 200:
            print(r.text[:2000], file=sys.stderr)
            print(f"HTTP {r.status_code}", file=sys.stderr)
            return 1

        body = r.json()
        outline = tensors_to_shapes(body)
        print(json.dumps(outline, indent=2, ensure_ascii=False))

        env = outline.get("denoising_env")
        dt = outline.get("dit_trajectory") or {}
        lat = (dt.get("latent_model_inputs") or {}) if isinstance(dt, dict) else {}
        lat_shape = lat.get("__tensor_shape__") if isinstance(lat, dict) else None
        if env and isinstance(env, dict):
            static = env.get("static") or {}
            pos = static.get("pos_cond_kwargs") or {}
            fc = pos.get("freqs_cis")
            if (
                isinstance(lat_shape, list)
                and len(lat_shape) >= 3
                and isinstance(fc, list)
                and len(fc) >= 1
                and isinstance(fc[0], dict)
            ):
                seq_lat = lat_shape[2]
                img_freq_shape = fc[0].get("__tensor_shape__")
                if (
                    isinstance(img_freq_shape, list)
                    and len(img_freq_shape) >= 1
                    and img_freq_shape[0] != seq_lat
                ):
                    print(
                        f"Shape check failed: pos freqs_cis image seq {img_freq_shape[0]} "
                        f"!= dit_trajectory.latent_model_inputs seq {seq_lat}",
                        file=sys.stderr,
                    )
                    return 1
        return 0
    finally:
        if not skip_launch:
            kill_server(proc)


if __name__ == "__main__":
    raise SystemExit(main())
