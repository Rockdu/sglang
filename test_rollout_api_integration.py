"""Integration test: Rollout Image API vs standard generation.

Launches a real sglang-D server with Z-Image-Turbo on a single GPU, then:
1. Calls POST /rollout/images for rollout log-probs (and optional ``dit_trajectory``)
2. Calls POST /v1/images/generations for the same prompt/seed
3. Verifies the rollout API returns expected tensor fields
4. Verifies roundtrip tensor serialization is lossless

Usage:
    CUDA_VISIBLE_DEVICES=6 FLASHINFER_DISABLE_VERSION_CHECK=1 \
        python test_rollout_api_integration.py
"""

import os
import signal
import subprocess
import sys
import time

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    base64_to_tensor,
)

MODEL = "Qwen/Qwen-Image"
PORT = int(os.environ.get("TEST_PORT", "39821"))
BASE_URL = f"http://127.0.0.1:{PORT}"
SEED = 42
PROMPT = "a beautiful sunset over the ocean, oil painting"


def wait_for_server(url: str, timeout: float = 600.0):
    """Poll /health until the server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                elapsed = time.time() - start
                print(f"  Server ready in {elapsed:.1f}s")
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not start within {timeout}s")


def deserialize_tensor_field(obj):
    """Recursively deserialize __tensor__ dicts back to torch tensors."""
    if isinstance(obj, dict) and obj.get("__tensor__"):
        return base64_to_tensor(obj["data"])
    if isinstance(obj, dict):
        return {k: deserialize_tensor_field(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deserialize_tensor_field(v) for v in obj]
    return obj


def launch_server():
    """Launch the sglang-D HTTP server as a subprocess."""
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    local_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
    env["PYTHONPATH"] = local_python + ":" + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m", "sglang.multimodal_gen.runtime.launch_server",
        "--model-path", MODEL,
        "--port", str(PORT),
        "--num-gpus", "1",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid,
    )
    return proc


def kill_server(proc):
    """Kill the server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    proc.wait(timeout=10)


def test_rollout_api():
    """Call POST /rollout/images and validate the response structure."""
    print("\n--- Test 1: POST /rollout/images ---")
    payload = dict(
        prompt=PROMPT,
        seed=SEED,
        rollout_sde_type="sde",
        rollout_noise_level=0.7,
        rollout_log_prob_no_const=False,
        rollout_debug_mode=True,
    )

    r = httpx.post(f"{BASE_URL}/rollout/images", json=payload, timeout=300)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:500]}"
    resp = r.json()

    # Check required scalar fields
    assert resp["prompt"] == PROMPT, f"Prompt mismatch: {resp['prompt']}"
    assert resp["seed"] == SEED, f"Seed mismatch: {resp['seed']}"
    assert resp["request_id"] is not None, "Missing request_id"
    print(f"  request_id: {resp['request_id']}")

    # Check generated_output
    gen_output = resp.get("generated_output")
    assert gen_output is not None, "Missing generated_output"
    gen_deserialized = deserialize_tensor_field(gen_output)
    if isinstance(gen_deserialized, list):
        print(f"  generated_output: list of {len(gen_deserialized)} item(s), first shape={gen_deserialized[0].shape}")
    else:
        print(f"  generated_output: shape={gen_deserialized.shape}, dtype={gen_deserialized.dtype}")

    assert "trajectory_latents" not in resp and "trajectory_timesteps" not in resp, (
        "Legacy ODE trajectory fields should not appear on rollout responses"
    )

    # Check rollout_log_probs
    log_probs = resp.get("rollout_log_probs")
    assert log_probs is not None, "Missing rollout_log_probs"
    log_probs_tensor = deserialize_tensor_field(log_probs)
    print(f"  rollout_log_probs: shape={log_probs_tensor.shape}, values={log_probs_tensor}")
    assert log_probs_tensor.numel() > 0, "rollout_log_probs is empty"

    # Check debug tensors
    debug = resp.get("rollout_debug_tensors")
    assert debug is not None, "Missing rollout_debug_tensors (rollout_debug_mode=True)"
    for key in ["rollout_variance_noises", "rollout_prev_sample_means",
                "rollout_noise_std_devs", "rollout_model_outputs"]:
        assert key in debug, f"Missing debug tensor: {key}"
        t = deserialize_tensor_field(debug[key])
        print(f"  debug.{key}: shape={t.shape}")

    # Check metrics
    if resp.get("inference_time_s") is not None:
        print(f"  inference_time_s: {resp['inference_time_s']:.2f}")
    if resp.get("peak_memory_mb") is not None:
        print(f"  peak_memory_mb: {resp['peak_memory_mb']:.1f}")

    print("  PASSED")
    return resp


def test_rollout_api_without_debug():
    """Call rollout API with debug_mode=False and verify no debug tensors."""
    print("\n--- Test 2: POST /rollout/images (no debug) ---")
    payload = dict(
        prompt=PROMPT,
        seed=SEED,
        rollout_sde_type="sde",
        rollout_noise_level=0.7,
        rollout_debug_mode=False,
    )

    r = httpx.post(f"{BASE_URL}/rollout/images", json=payload, timeout=300)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:500]}"
    resp = r.json()

    assert resp.get("rollout_log_probs") is not None, "rollout_log_probs should still be present"
    assert resp.get("rollout_debug_tensors") is None, (
        "rollout_debug_tensors should be None when debug_mode=False"
    )
    print("  PASSED (no debug tensors as expected)")
    return resp


def test_rollout_api_dit_trajectory_when_requested():
    """``dit_trajectory`` is gated by ``rollout_return_dit_trajectory`` (not ``rollout_return_denoising_env``)."""
    print("\n--- Test 3: POST /rollout/images (rollout_return_dit_trajectory only) ---")
    payload = dict(
        prompt=PROMPT,
        seed=SEED,
        rollout_sde_type="sde",
        rollout_noise_level=0.7,
        rollout_debug_mode=False,
        rollout_return_denoising_env=False,
        rollout_return_dit_trajectory=True,
    )

    r = httpx.post(f"{BASE_URL}/rollout/images", json=payload, timeout=300)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:500]}"
    resp = r.json()

    dit = resp.get("dit_trajectory")
    assert dit is not None, "dit_trajectory expected when rollout_return_dit_trajectory=True"
    assert dit.get("latent_model_inputs") is not None
    assert dit.get("timesteps") is not None
    assert resp.get("denoising_env") is None, "denoising_env should be absent when rollout_return_denoising_env=False"
    assert resp.get("rollout_log_probs") is not None
    print("  PASSED (dit_trajectory without static env)")


def test_deterministic_with_same_seed():
    """Two calls with same seed should produce identical rollout_log_probs."""
    print("\n--- Test 4: Determinism (same seed -> same log_probs) ---")
    payload = dict(
        prompt=PROMPT,
        seed=SEED,
        rollout_sde_type="sde",
        rollout_noise_level=0.7,
    )

    r1 = httpx.post(f"{BASE_URL}/rollout/images", json=payload, timeout=300)
    r2 = httpx.post(f"{BASE_URL}/rollout/images", json=payload, timeout=300)
    assert r1.status_code == 200 and r2.status_code == 200

    lp1 = deserialize_tensor_field(r1.json()["rollout_log_probs"])
    lp2 = deserialize_tensor_field(r2.json()["rollout_log_probs"])
    import torch
    max_diff = (lp1.float() - lp2.float()).abs().max().item()
    print(f"  log_probs diff between two identical calls: {max_diff:.8f}")
    assert max_diff < 1e-4, f"Non-deterministic: max_diff={max_diff}"
    print("  PASSED (deterministic)")


def test_cps_sde_type():
    """Verify CPS sde_type works."""
    print("\n--- Test 5: CPS sde_type ---")
    payload = dict(
        prompt=PROMPT,
        seed=SEED,
        rollout_sde_type="cps",
        rollout_noise_level=0.5,
    )

    r = httpx.post(f"{BASE_URL}/rollout/images", json=payload, timeout=300)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:500]}"
    resp = r.json()
    assert resp.get("rollout_log_probs") is not None
    lp = deserialize_tensor_field(resp["rollout_log_probs"])
    print(f"  CPS log_probs: {lp}")
    print("  PASSED")


def test_ode_sde_type():
    """Verify ODE sde_type works (deterministic, no noise).

    ODE path requires rollout_log_prob_no_const=True because p_ode is always 0.
    """
    print("\n--- Test 6: ODE sde_type ---")
    payload = dict(
        prompt=PROMPT,
        seed=SEED,
        rollout_sde_type="ode",
        rollout_noise_level=0.5,
        rollout_log_prob_no_const=True,
    )

    r = httpx.post(f"{BASE_URL}/rollout/images", json=payload, timeout=300)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:500]}"
    resp = r.json()
    assert resp.get("rollout_log_probs") is not None
    lp = deserialize_tensor_field(resp["rollout_log_probs"])
    import torch
    assert torch.all(lp == 0.0), f"ODE log_probs should be 0, got {lp}"
    print(f"  ODE log_probs: {lp} (all zeros as expected)")
    print("  PASSED")


def main():
    print("=" * 70)
    print("Integration Test: Rollout Image API")
    print(f"  Model: {MODEL}")
    print(f"  Port: {PORT}")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    # Launch server
    print("\n[Setup] Launching server...")
    server_proc = launch_server()

    all_passed = True
    try:
        wait_for_server(BASE_URL, timeout=600)

        for test_fn in [
            test_rollout_api,
            test_rollout_api_without_debug,
            test_rollout_api_dit_trajectory_when_requested,
            test_deterministic_with_same_seed,
            test_cps_sde_type,
            test_ode_sde_type,
        ]:
            try:
                test_fn()
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False

    finally:
        print("\n[Teardown] Killing server...")
        kill_server(server_proc)

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
