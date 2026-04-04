"""Integration: ``POST /rollout/images`` (JSON array per sample) + rollout vs baseline perf.

Paired A/B with same seed per pair and alternating order. See argparse for GPUs / ports.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

import httpx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import _maybe_deserialize
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
            r = httpx.get(f"{base_url}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server not ready within {timeout_s}s: {base_url}")


def _launch_server(
    model_path: str,
    port: int,
    report_lines: list[str],
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
    cutlass_pkg_path = (
        "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages"
    )
    if os.path.isdir(cutlass_pkg_path):
        py_paths.append(cutlass_pkg_path)
    prev = env.get("PYTHONPATH", "")
    if prev:
        py_paths.append(prev)
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
    report_lines.append("")
    report_lines.append(f"[launch] {' '.join(cmd)}")
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


def _validate_tensor_payload(
    resp_json: list[dict[str, Any]],
    *,
    expect_denoising_env: bool,
    expect_dit_trajectory: bool,
) -> None:
    assert isinstance(resp_json, list) and len(resp_json) >= 1
    for sample in resp_json:
        gen = sample.get("generated_output")
        assert gen is not None
        assert _count_tensors(_maybe_deserialize(gen)) > 0
        if expect_denoising_env:
            denv = sample.get("denoising_env")
            assert denv is not None
            d = _maybe_deserialize(denv)
            assert isinstance(d, dict) and any(
                d.get(k) is not None
                for k in ("image_kwargs", "pos_cond_kwargs", "neg_cond_kwargs", "guidance")
            )
            assert "trajectory" not in d
        else:
            assert sample.get("denoising_env") is None
        if expect_dit_trajectory:
            dt = sample.get("dit_trajectory")
            assert dt is not None
            lmi = _maybe_deserialize(dt["latent_model_inputs"])
            ts_raw = dt.get("timesteps")
            if ts_raw is not None:
                ts = _maybe_deserialize(ts_raw)
                assert isinstance(lmi, torch.Tensor) and isinstance(ts, torch.Tensor)
                assert lmi.shape[0] == ts.shape[0]
            else:
                assert isinstance(lmi, torch.Tensor)
        else:
            assert sample.get("dit_trajectory") is None


def _run_rollout_request(
    base_url: str,
    prompt: str,
    seed: int,
    rollout: bool,
    rollout_return_denoising_env: bool,
    rollout_return_dit_trajectory: bool,
) -> tuple[float, float, list[dict[str, Any]]]:
    payload = {
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": 8,
        "guidance_scale": 4.0,
        "rollout_sde_type": "sde",
        "rollout_noise_level": 0.7,
        "rollout_debug_mode": False,
        "rollout_return_denoising_env": rollout_return_denoising_env,
        "rollout_return_dit_trajectory": rollout_return_dit_trajectory,
        "extra_sampling_params": {"rollout": rollout},
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{base_url}/rollout/images", json=payload, timeout=600)
    wall_s = time.perf_counter() - t0
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:500]}"
    body = r.json()
    assert isinstance(body, list), body
    infer_s = float(body[0].get("inference_time_s") or wall_s)
    return wall_s, infer_s, body


def _warmup_both_scenarios(
    base_url: str, prompt: str, base_seed: int, warmup_runs: int, report_lines: list[str]
) -> None:
    report_lines.append("")
    report_lines.append("[warmup] both code paths")
    for i in range(warmup_runs):
        s = base_seed + i
        wall_s, infer_s, jr = _run_rollout_request(
            base_url, prompt, s, True, True, True
        )
        _validate_tensor_payload(jr, expect_denoising_env=True, expect_dit_trajectory=True)
        report_lines.append(f"  warmup-{i + 1}a rollout+env: wall={wall_s:.3f}s infer={infer_s:.3f}s")
        wall_s, infer_s, jb = _run_rollout_request(
            base_url, prompt, s + 50_000, False, False, False
        )
        _validate_tensor_payload(jb, expect_denoising_env=False, expect_dit_trajectory=False)
        report_lines.append(f"  warmup-{i + 1}b baseline:    wall={wall_s:.3f}s infer={infer_s:.3f}s")


def _benchmark_paired(
    base_url: str,
    prompt: str,
    measure_seed_base: int,
    measured_runs: int,
    report_lines: list[str],
) -> tuple[ScenarioResult, ScenarioResult]:
    wall_roll, infer_roll = [], []
    wall_base, infer_base = [], []
    report_lines.append("")
    report_lines.append("[measured] paired runs (same seed per pair, order alternates)")
    for i in range(measured_runs):
        s = measure_seed_base + i
        roll_first = i % 2 == 0
        if roll_first:
            wr, ir, jr = _run_rollout_request(base_url, prompt, s, True, True, True)
            _validate_tensor_payload(jr, expect_denoising_env=True, expect_dit_trajectory=True)
            wb, ib, jb = _run_rollout_request(base_url, prompt, s, False, False, False)
            _validate_tensor_payload(jb, expect_denoising_env=False, expect_dit_trajectory=False)
        else:
            wb, ib, jb = _run_rollout_request(base_url, prompt, s, False, False, False)
            _validate_tensor_payload(jb, expect_denoising_env=False, expect_dit_trajectory=False)
            wr, ir, jr = _run_rollout_request(base_url, prompt, s, True, True, True)
            _validate_tensor_payload(jr, expect_denoising_env=True, expect_dit_trajectory=True)
        wall_roll.append(wr)
        infer_roll.append(ir)
        wall_base.append(wb)
        infer_base.append(ib)
        order = "rollout→baseline" if roll_first else "baseline→rollout"
        report_lines.append(
            f"  pair-{i + 1} seed={s} ({order}): "
            f"rollout wall={wr:.3f}s infer={ir:.3f}s | baseline wall={wb:.3f}s infer={ib:.3f}s"
        )
    return (
        ScenarioResult(
            name="rollout+env+dit_trajectory",
            wall_times_s=wall_roll,
            inference_times_s=infer_roll,
        ),
        ScenarioResult(
            name="baseline",
            wall_times_s=wall_base,
            inference_times_s=infer_base,
        ),
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
) -> list[str]:
    lines: list[str] = []
    base_url = f"http://127.0.0.1:{port}"
    lines += ["", "=" * 90, f"[model] {model_name}", "=" * 90]
    proc = _launch_server(
        model_name,
        port,
        lines,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_degree=sp_degree,
        enable_cfg_parallel=enable_cfg_parallel,
    )
    try:
        _wait_for_server(base_url)
        _warmup_both_scenarios(base_url, prompt, seed, warmup_runs, lines)
        with_r, without = _benchmark_paired(
            base_url, prompt, seed + 1000, measured_runs, lines
        )
        wr = with_r.wall_avg_s / max(without.wall_avg_s, 1e-6)
        ir = with_r.infer_avg_s / max(without.infer_avg_s, 1e-6)
        lines += [
            "",
            "[result]",
            "  wall = HTTP round-trip; infer = server pipeline (inference_time_s).",
            f"  baseline wall avg: {without.wall_avg_s:.3f}s"
            + (f" (pstdev {pstdev(without.wall_times_s):.3f}s)" if measured_runs > 1 else ""),
            f"  rollout wall avg: {with_r.wall_avg_s:.3f}s"
            + (f" (pstdev {pstdev(with_r.wall_times_s):.3f}s)" if measured_runs > 1 else ""),
            f"  wall ratio: {wr:.3f}x",
            f"  baseline infer avg: {without.infer_avg_s:.3f}s"
            + (f" (pstdev {pstdev(without.inference_times_s):.3f}s)" if measured_runs > 1 else ""),
            f"  rollout infer avg: {with_r.infer_avg_s:.3f}s"
            + (f" (pstdev {pstdev(with_r.inference_times_s):.3f}s)" if measured_runs > 1 else ""),
            f"  infer ratio: {ir:.3f}x",
        ]
        if measured_runs < 5:
            lines.append(f"  Hint: --measured-runs={measured_runs} is low; small ratios may be noise.")
        return lines
    finally:
        _kill_server(proc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=39911)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--measured-runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-gpus", type=int, default=int(os.environ.get("TEST_NUM_GPUS", "1")))
    p.add_argument("--tp-size", type=int, default=None)
    p.add_argument("--sp-degree", type=int, default=None)
    p.add_argument("--enable-cfg-parallel", "--cfgp", action="store_true")
    p.add_argument(
        "--prompt",
        default="a cinematic portrait of an astronaut walking in neon rain, ultra detailed",
    )
    args = p.parse_args()
    kw = dict(
        num_gpus=args.num_gpus,
        tp_size=args.tp_size,
        sp_degree=args.sp_degree,
        enable_cfg_parallel=args.enable_cfg_parallel,
    )
    report = [
        "=" * 90,
        "rollout_api perf — full report",
        "=" * 90,
    ]
    report.extend(
        _run_model_suite(
            DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
            args.port,
            args.prompt,
            args.seed,
            args.warmup_runs,
            args.measured_runs,
            **kw,
        )
    )
    report.extend(
        _run_model_suite(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            args.port + 1,
            args.prompt,
            args.seed + 10_000,
            args.warmup_runs,
            args.measured_runs,
            **kw,
        )
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
