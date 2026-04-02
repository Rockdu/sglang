"""Integration test: rollout_api tensor retrieval and perf comparison.

This script validates two things through ``POST /rollout/images``:
1. Returned tensor payloads can be deserialized correctly.
2. Inference latency difference between:
   - rollout enabled + DiT env return enabled
   - rollout disabled + DiT env return disabled
   on both Qwen-Image and Z-Image-Turbo.

Performance methodology: after warming **both** code paths, each measured **pair** uses the
**same diffusion seed** for rollout+env vs baseline, and **alternates call order** (AB / BA) so
neither arm always benefits from running immediately after the other. Older versions compared
different seeds and always ran rollout first, which could show spurious speedups/slowdowns.

Usage:
    CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        python test_rollout_api_perf_compare_qwen_zimage.py

    CUDA_VISIBLE_DEVICES=0,1 ... python test_rollout_api_perf_compare_qwen_zimage.py --num-gpus 2
    CUDA_VISIBLE_DEVICES=0,1,2 ... python test_rollout_api_perf_compare_qwen_zimage.py --num-gpus 3 \\
        --tp-size 1 --sp-degree 3

    Optional: --tp-size, --sp-degree, --enable-cfg-parallel / --cfgp (forwarded to launch_server).

    Benchmark lines (model headers, launch cmd, warmup, pairs, [result]) are buffered and printed
    once after both models finish. Server process logs still go to stdout/stderr in real time.
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
    resp_json: dict[str, Any],
    *,
    expect_denoising_env: bool,
    expect_dit_trajectory: bool,
) -> None:
    generated = resp_json.get("generated_output")
    assert generated is not None, "generated_output missing"
    generated_obj = _maybe_deserialize(generated)
    assert _count_tensors(generated_obj) > 0, "generated_output has no tensor payload"

    if expect_denoising_env:
        denv = resp_json.get("denoising_env")
        assert denv is not None, "denoising_env should exist when rollout_return_denoising_env=True"
        denv_obj = _maybe_deserialize(denv)
        assert isinstance(denv_obj, dict), "denoising_env should deserialize to a dict"
        assert any(
            denv_obj.get(k) is not None
            for k in ("image_kwargs", "pos_cond_kwargs", "neg_cond_kwargs", "guidance")
        ), "denoising_env should contain at least one conditioning field"
        assert "trajectory" not in denv_obj, "trajectory should not be nested under denoising_env"
    else:
        assert resp_json.get("denoising_env") is None, (
            "denoising_env should be None when rollout_return_denoising_env=False"
        )

    if expect_dit_trajectory:
        dt_raw = resp_json.get("dit_trajectory")
        assert dt_raw is not None, "dit_trajectory missing"
        lmi_raw = dt_raw.get("latent_model_inputs")
        tsteps_raw = dt_raw.get("timesteps")
        assert lmi_raw is not None, "dit_trajectory.latent_model_inputs missing"
        assert tsteps_raw is not None, "dit_trajectory.timesteps missing"
        lmi = _maybe_deserialize(lmi_raw)
        tsteps = _maybe_deserialize(tsteps_raw)
        assert isinstance(lmi, torch.Tensor), "dit_trajectory.latent_model_inputs not tensor after decode"
        assert isinstance(tsteps, torch.Tensor), "dit_trajectory.timesteps not tensor after decode"
        assert lmi.shape[1] == tsteps.shape[0], "step count mismatch (DiT trajectory)"
    else:
        assert resp_json.get("dit_trajectory") is None, (
            "dit_trajectory should be None when rollout_return_dit_trajectory=False"
        )


def _run_rollout_request(
    base_url: str,
    prompt: str,
    seed: int,
    rollout: bool,
    rollout_return_denoising_env: bool,
    rollout_return_dit_trajectory: bool,
) -> tuple[float, float, dict[str, Any]]:
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


def _warmup_both_scenarios(
    base_url: str,
    prompt: str,
    base_seed: int,
    warmup_runs: int,
    report_lines: list[str],
) -> None:
    """Warm up rollout-on and rollout-off paths; seeds only need to differ between calls."""
    report_lines.append("")
    report_lines.append("[warmup] both code paths")
    for i in range(warmup_runs):
        s = base_seed + i
        wall_s, infer_s, resp_json = _run_rollout_request(
            base_url=base_url,
            prompt=prompt,
            seed=s,
            rollout=True,
            rollout_return_denoising_env=True,
            rollout_return_dit_trajectory=True,
        )
        _validate_tensor_payload(
            resp_json,
            expect_denoising_env=True,
            expect_dit_trajectory=True,
        )
        report_lines.append(
            f"  warmup-{i + 1}a rollout+env: wall={wall_s:.3f}s infer={infer_s:.3f}s"
        )
        wall_s, infer_s, resp_json = _run_rollout_request(
            base_url=base_url,
            prompt=prompt,
            seed=s + 50_000,
            rollout=False,
            rollout_return_denoising_env=False,
            rollout_return_dit_trajectory=False,
        )
        _validate_tensor_payload(
            resp_json,
            expect_denoising_env=False,
            expect_dit_trajectory=False,
        )
        report_lines.append(
            f"  warmup-{i + 1}b baseline:    wall={wall_s:.3f}s infer={infer_s:.3f}s"
        )


def _benchmark_paired(
    base_url: str,
    prompt: str,
    measure_seed_base: int,
    measured_runs: int,
    report_lines: list[str],
) -> tuple[ScenarioResult, ScenarioResult]:
    """Fair A/B: same diffusion seed per pair, alternate which scenario runs first (AB/BA).

    Avoids (1) different RNG trajectories per arm and (2) always warming the GPU for the same
    arm first. Client *wall* still includes JSON/body transfer; server *infer* follows
    ``OutputBatch.metrics.total_duration_s`` (serialization of huge base64 is not in infer).
    """
    wall_roll, infer_roll = [], []
    wall_base, infer_base = [], []
    report_lines.append("")
    report_lines.append("[measured] paired runs (same seed per pair, order alternates)")
    for i in range(measured_runs):
        s = measure_seed_base + i
        roll_first = i % 2 == 0
        if roll_first:
            wr, ir, jr = _run_rollout_request(
                base_url,
                prompt,
                s,
                rollout=True,
                rollout_return_denoising_env=True,
                rollout_return_dit_trajectory=True,
            )
            _validate_tensor_payload(
                jr,
                expect_denoising_env=True,
                expect_dit_trajectory=True,
            )
            wb, ib, jb = _run_rollout_request(
                base_url,
                prompt,
                s,
                rollout=False,
                rollout_return_denoising_env=False,
                rollout_return_dit_trajectory=False,
            )
            _validate_tensor_payload(
                jb,
                expect_denoising_env=False,
                expect_dit_trajectory=False,
            )
        else:
            wb, ib, jb = _run_rollout_request(
                base_url,
                prompt,
                s,
                rollout=False,
                rollout_return_denoising_env=False,
                rollout_return_dit_trajectory=False,
            )
            _validate_tensor_payload(
                jb,
                expect_denoising_env=False,
                expect_dit_trajectory=False,
            )
            wr, ir, jr = _run_rollout_request(
                base_url,
                prompt,
                s,
                rollout=True,
                rollout_return_denoising_env=True,
                rollout_return_dit_trajectory=True,
            )
            _validate_tensor_payload(
                jr,
                expect_denoising_env=True,
                expect_dit_trajectory=True,
            )
        wall_roll.append(wr)
        infer_roll.append(ir)
        wall_base.append(wb)
        infer_base.append(ib)
        order = "rollout→baseline" if roll_first else "baseline→rollout"
        report_lines.append(
            f"  pair-{i + 1} seed={s} ({order}): "
            f"rollout wall={wr:.3f}s infer={ir:.3f}s | "
            f"baseline wall={wb:.3f}s infer={ib:.3f}s"
        )

    return (
        ScenarioResult(
            name="rollout=on + rollout_return_denoising_env=on + rollout_return_dit_trajectory=on",
            wall_times_s=wall_roll,
            inference_times_s=infer_roll,
        ),
        ScenarioResult(
            name="rollout=off + dit capture off",
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
    lines.append("")
    lines.append("=" * 90)
    lines.append(f"[model] {model_name}")
    lines.append("=" * 90)

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
        _warmup_both_scenarios(
            base_url=base_url,
            prompt=prompt,
            base_seed=seed,
            warmup_runs=warmup_runs,
            report_lines=lines,
        )
        with_rollout, without_rollout = _benchmark_paired(
            base_url=base_url,
            prompt=prompt,
            measure_seed_base=seed + 1000,
            measured_runs=measured_runs,
            report_lines=lines,
        )

        wall_ratio = with_rollout.wall_avg_s / max(without_rollout.wall_avg_s, 1e-6)
        infer_ratio = with_rollout.infer_avg_s / max(without_rollout.infer_avg_s, 1e-6)
        lines.append("")
        lines.append("[result]")
        lines.append(
            "  Note: wall = full HTTP round-trip (larger JSON when dit capture flags on); "
            "infer = server pipeline duration only (see rollout_api inference_time_s)."
        )
        bw = f"  baseline wall avg: {without_rollout.wall_avg_s:.3f}s"
        if measured_runs > 1:
            bw += f" (pstdev {pstdev(without_rollout.wall_times_s):.3f}s)"
        lines.append(bw)
        rw = f"  rollout+env wall avg: {with_rollout.wall_avg_s:.3f}s"
        if measured_runs > 1:
            rw += f" (pstdev {pstdev(with_rollout.wall_times_s):.3f}s)"
        lines.append(rw)
        lines.append(f"  wall slowdown ratio: {wall_ratio:.3f}x")
        bi = f"  baseline infer avg: {without_rollout.infer_avg_s:.3f}s"
        if measured_runs > 1:
            bi += f" (pstdev {pstdev(without_rollout.inference_times_s):.3f}s)"
        lines.append(bi)
        ri = f"  rollout+env infer avg: {with_rollout.infer_avg_s:.3f}s"
        if measured_runs > 1:
            ri += f" (pstdev {pstdev(with_rollout.inference_times_s):.3f}s)"
        lines.append(ri)
        lines.append(f"  infer slowdown ratio: {infer_ratio:.3f}x")
        if measured_runs < 5:
            lines.append(
                f"  Hint: --measured-runs={measured_runs} is low; "
                "ratios within ~1–2% are often noise."
            )
        return lines
    finally:
        _kill_server(proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=39911)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=10)
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

    # Run sequentially to avoid multi-model VRAM contention; print one report at the end.
    report: list[str] = [
        "=" * 90,
        "rollout_api perf comparison — full report (after all model runs complete)",
        "=" * 90,
    ]
    report.extend(
        _run_model_suite(
            model_name=DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
            port=args.port,
            prompt=args.prompt,
            seed=args.seed,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            **suite_kw,
        )
    )
    report.extend(
        _run_model_suite(
            model_name=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,  # Tongyi-MAI/Z-Image-Turbo
            port=args.port + 1,
            prompt=args.prompt,
            seed=args.seed + 10_000,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            **suite_kw,
        )
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
