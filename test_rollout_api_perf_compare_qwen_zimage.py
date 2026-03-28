"""Integration test: rollout_api tensor retrieval and perf comparison.

This script validates two things through ``POST /rollout/images``:
1. Returned tensor payloads can be deserialized correctly.
2. Inference latency difference between:
   - rollout enabled + DiT env return enabled
   - rollout disabled + DiT env return disabled
   on both Qwen-Image and Z-Image-Turbo.

Usage:
    CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        python test_rollout_api_perf_compare_qwen_zimage.py

    CUDA_VISIBLE_DEVICES=0,1 ... python test_rollout_api_perf_compare_qwen_zimage.py --num-gpus 2
    CUDA_VISIBLE_DEVICES=0,1,2 ... python test_rollout_api_perf_compare_qwen_zimage.py --num-gpus 3 \\
        --tp-size 1 --sp-degree 3

    Optional: --tp-size, --sp-degree, --enable-cfg-parallel / --cfgp (forwarded to launch_server).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

import httpx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_deserialize,
)
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)


@dataclass
class ScenarioResult:
    name: str
    wall_times_s: list[float]
    inference_times_s: list[float]

    @property
    def wall_avg_s(self) -> float:
        return mean(self.wall_times_s)

    @property
    def infer_avg_s(self) -> float:
        return mean(self.inference_times_s)


def _wait_for_server(base_url: str, timeout_s: float = 900.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server not ready within {timeout_s}s: {base_url}")


def _launch_server(
    model_path: str,
    port: int,
    *,
    num_gpus: int = 1,
    tp_size: int | None = None,
    sp_degree: int | None = None,
    enable_cfg_parallel: bool = False,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    local_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
    py_paths = [local_python]
    # nvidia-cutlass-dsl installs `cutlass` under this nested path, not default sys.path.
    cutlass_pkg_path = (
        "/usr/local/lib/python3.12/dist-packages/"
        "nvidia_cutlass_dsl/python_packages"
    )
    if os.path.isdir(cutlass_pkg_path):
        py_paths.append(cutlass_pkg_path)
    prev_py_path = env.get("PYTHONPATH", "")
    if prev_py_path:
        py_paths.append(prev_py_path)
    env["PYTHONPATH"] = ":".join(py_paths)

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
    print(f"\n[launch] {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid,
    )


def _kill_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def _count_tensors(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return 1
    if isinstance(obj, dict):
        return sum(_count_tensors(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_tensors(v) for v in obj)
    return 0


def _validate_tensor_payload(resp_json: dict[str, Any], expect_dit_env: bool) -> None:
    generated = resp_json.get("generated_output")
    assert generated is not None, "generated_output missing"
    generated_obj = _maybe_deserialize(generated)
    assert _count_tensors(generated_obj) > 0, "generated_output has no tensor payload"

    if expect_dit_env:
        denv = resp_json.get("denoising_env")
        assert denv is not None, "denoising_env should exist when return_dit_env=True"
        denv_obj = _maybe_deserialize(denv)
        assert denv_obj.get("static") is not None, "denoising_env.static missing"
        traj = denv_obj.get("trajectory")
        assert traj is not None, "denoising_env.trajectory missing"
        lmi = traj.get("latent_model_inputs")
        tsteps = traj.get("timesteps")
        assert isinstance(lmi, torch.Tensor), "latent_model_inputs not tensor after decode"
        assert isinstance(tsteps, torch.Tensor), "timesteps not tensor after decode"
        # latent_model_inputs [B, T, ...]; timesteps [T] shared across batch
        assert lmi.shape[1] == tsteps.shape[0], "step count mismatch in denoising_env"
    else:
        assert resp_json.get("denoising_env") is None, (
            "denoising_env should be None when return_dit_env=False"
        )


def _run_rollout_request(
    base_url: str,
    prompt: str,
    seed: int,
    rollout: bool,
    return_dit_env: bool,
) -> tuple[float, float, dict[str, Any]]:
    payload = {
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": 8,
        "guidance_scale": 4.0,
        "rollout_sde_type": "sde",
        "rollout_noise_level": 0.7,
        "rollout_debug_mode": False,
        "return_trajectory_latents": False,
        "return_trajectory_decoded": False,
        "return_dit_env": return_dit_env,
        # rollout_api internally forces rollout=True, so we override explicitly here.
        "extra_sampling_params": {"rollout": rollout},
    }

    start = time.perf_counter()
    r = httpx.post(f"{base_url}/rollout/images", json=payload, timeout=600)
    wall_s = time.perf_counter() - start
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:500]}"
    resp_json = r.json()
    infer_s = float(resp_json.get("inference_time_s") or wall_s)
    return wall_s, infer_s, resp_json


def _benchmark_scenario(
    base_url: str,
    prompt: str,
    seed: int,
    scenario_name: str,
    rollout: bool,
    return_dit_env: bool,
    warmup_runs: int,
    measured_runs: int,
) -> ScenarioResult:
    print(f"\n[scenario] {scenario_name}")
    for i in range(warmup_runs):
        wall_s, infer_s, resp_json = _run_rollout_request(
            base_url=base_url,
            prompt=prompt,
            seed=seed + i,
            rollout=rollout,
            return_dit_env=return_dit_env,
        )
        _validate_tensor_payload(resp_json, expect_dit_env=return_dit_env)
        print(f"  warmup-{i + 1}: wall={wall_s:.3f}s infer={infer_s:.3f}s")

    wall_times: list[float] = []
    infer_times: list[float] = []
    for i in range(measured_runs):
        wall_s, infer_s, resp_json = _run_rollout_request(
            base_url=base_url,
            prompt=prompt,
            seed=seed + 100 + i,
            rollout=rollout,
            return_dit_env=return_dit_env,
        )
        _validate_tensor_payload(resp_json, expect_dit_env=return_dit_env)
        wall_times.append(wall_s)
        infer_times.append(infer_s)
        print(f"  run-{i + 1}:    wall={wall_s:.3f}s infer={infer_s:.3f}s")

    return ScenarioResult(
        name=scenario_name,
        wall_times_s=wall_times,
        inference_times_s=infer_times,
    )


def _run_model_suite(
    model_name: str,
    port: int,
    prompt: str,
    seed: int,
    warmup_runs: int,
    measured_runs: int,
    *,
    num_gpus: int = 1,
    tp_size: int | None = None,
    sp_degree: int | None = None,
    enable_cfg_parallel: bool = False,
) -> None:
    base_url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 90)
    print(f"[model] {model_name}")
    print("=" * 90)

    proc = _launch_server(
        model_name,
        port,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_degree=sp_degree,
        enable_cfg_parallel=enable_cfg_parallel,
    )
    try:
        _wait_for_server(base_url)
        with_rollout = _benchmark_scenario(
            base_url=base_url,
            prompt=prompt,
            seed=seed,
            scenario_name="rollout=on + return_dit_env=on",
            rollout=True,
            return_dit_env=True,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        without_rollout = _benchmark_scenario(
            base_url=base_url,
            prompt=prompt,
            seed=seed + 1000,
            scenario_name="rollout=off + return_dit_env=off",
            rollout=False,
            return_dit_env=False,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )

        wall_ratio = with_rollout.wall_avg_s / max(without_rollout.wall_avg_s, 1e-6)
        infer_ratio = with_rollout.infer_avg_s / max(without_rollout.infer_avg_s, 1e-6)
        print("\n[result]")
        print(f"  baseline wall avg: {without_rollout.wall_avg_s:.3f}s")
        print(f"  rollout+env wall avg: {with_rollout.wall_avg_s:.3f}s")
        print(f"  wall slowdown ratio: {wall_ratio:.3f}x")
        print(f"  baseline infer avg: {without_rollout.infer_avg_s:.3f}s")
        print(f"  rollout+env infer avg: {with_rollout.infer_avg_s:.3f}s")
        print(f"  infer slowdown ratio: {infer_ratio:.3f}x")
    finally:
        _kill_server(proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=39911)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("TEST_NUM_GPUS", "1")),
        help="launch_server --num-gpus (set CUDA_VISIBLE_DEVICES accordingly).",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size; omit for server default (1).",
    )
    parser.add_argument(
        "--sp-degree",
        type=int,
        default=None,
        help="Sequence parallel degree; omit for server auto from remaining GPUs.",
    )
    parser.add_argument(
        "--enable-cfg-parallel",
        "--cfgp",
        action="store_true",
        help="launch_server --enable-cfg-parallel",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cinematic portrait of an astronaut walking in neon rain, ultra detailed",
    )
    args = parser.parse_args()

    suite_kw = dict(
        num_gpus=args.num_gpus,
        tp_size=args.tp_size,
        sp_degree=args.sp_degree,
        enable_cfg_parallel=args.enable_cfg_parallel,
    )

    # Run sequentially to avoid multi-model VRAM contention.
    _run_model_suite(
        model_name=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
        port=args.port,
        prompt=args.prompt,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        **suite_kw,
    )
    _run_model_suite(
        model_name=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,  # Tongyi-MAI/Z-Image-Turbo
        port=args.port + 1,
        prompt=args.prompt,
        seed=args.seed + 10_000,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        **suite_kw,
    )


if __name__ == "__main__":
    main()
