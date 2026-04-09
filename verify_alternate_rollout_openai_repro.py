#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-off check: does interleaving rollout and non-rollout HTTP requests change a probe request?

Uses fixed JSON bodies and compares stable fingerprints so the script is reproducible given a
deterministic server (same model, same CUDA/cuDNN settings, same code).

Requires a running multimodal_gen HTTP server (default base URL ``http://127.0.0.1:30000``).

Example::

    python verify_alternate_rollout_openai_repro.py --base-url http://127.0.0.1:30000

Exit code 0 if all probe fingerprints match their isolated baseline; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from typing import Any

import requests


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, tuple):
        return [_canonicalize(x) for x in obj]
    return obj


def _canonical_json_sha256(obj: Any) -> str:
    blob = json.dumps(
        _canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _post_json(base_url: str, path: str, body: dict, timeout_s: float) -> dict:
    url = base_url.rstrip("/") + path
    r = requests.post(url, json=body, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _rollout_fingerprints(row: dict) -> tuple[str, str]:
    """(generated_output only, rollout_log_probs only) — both exclude request metadata."""
    meta = {"request_id", "inference_time_s", "peak_memory_mb"}
    slim = {k: v for k, v in row.items() if k not in meta}
    gen = slim.get("generated_output")
    lp = slim.get("rollout_log_probs")
    return _canonical_json_sha256(gen), _canonical_json_sha256(lp)


def _openai_image_fingerprint(resp: dict) -> str:
    data = resp.get("data") or []
    if not data:
        raise ValueError("OpenAI image response has empty data")
    b64 = data[0].get("b64_json")
    if not b64:
        raise ValueError("Expected response_format=b64_json with b64_json field")
    raw = base64.b64decode(b64)
    return hashlib.sha256(raw).hexdigest()


def _assert_all_equal(name: str, values: list[str]) -> None:
    if len(set(values)) != 1:
        print(f"{name}: MISMATCH among {len(values)} runs", file=sys.stderr)
        for i, v in enumerate(values):
            print(f"  [{i}] {v}", file=sys.stderr)
        raise SystemExit(2)
    print(f"{name}: OK (all {len(values)} digests identical)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--prompt", default="a red circle on white background, flat vector")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--true-cfg-scale", type=float, default=None)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--generator-device", default="cuda")
    p.add_argument("--baseline-repeats", type=int, default=3, help="Repeated identical probes (sanity).")
    p.add_argument(
        "--alternate-rounds",
        type=int,
        default=6,
        help="Each round: one filler rollout + one filler openai, then probe.",
    )
    args = p.parse_args()

    probe_rollout_body: dict[str, Any] = {
        "prompt": args.prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "num_outputs_per_prompt": 1,
        "guidance_scale": args.guidance_scale,
        "generator_device": args.generator_device,
        "rollout_sde_type": "sde",
        "rollout_noise_level": 0.7,
        "rollout_log_prob_no_const": False,
        "rollout_debug_mode": False,
    }
    if args.negative_prompt is not None:
        probe_rollout_body["negative_prompt"] = args.negative_prompt
    if args.true_cfg_scale is not None:
        probe_rollout_body["true_cfg_scale"] = args.true_cfg_scale

    probe_openai_body: dict[str, Any] = {
        "prompt": args.prompt,
        "seed": args.seed,
        "n": 1,
        "response_format": "b64_json",
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator_device": args.generator_device,
    }
    if args.negative_prompt is not None:
        probe_openai_body["negative_prompt"] = args.negative_prompt
    if args.true_cfg_scale is not None:
        probe_openai_body["true_cfg_scale"] = args.true_cfg_scale

    # Fillers must differ from the probe so they exercise the scheduler without colliding on intent.
    filler_rollout_body = {**probe_rollout_body, "prompt": args.prompt + " | filler rollout", "seed": args.seed + 1_000_003}
    filler_openai_body = {**probe_openai_body, "prompt": args.prompt + " | filler openai", "seed": args.seed + 2_000_005}

    bu = args.base_url
    t = args.timeout_s

    # --- Rollout probe: isolated baseline ---
    roll_gen_digests: list[str] = []
    roll_lp_digests: list[str] = []
    for _ in range(args.baseline_repeats):
        rows = _post_json(bu, "/rollout/generate", probe_rollout_body, t)
        if not isinstance(rows, list) or not rows:
            raise SystemExit(f"Unexpected rollout response: {rows!r}")
        g, lp = _rollout_fingerprints(rows[0])
        roll_gen_digests.append(g)
        roll_lp_digests.append(lp)
    _assert_all_equal("rollout probe generated_output", roll_gen_digests)
    _assert_all_equal("rollout probe rollout_log_probs", roll_lp_digests)
    ref_roll_g, ref_roll_lp = roll_gen_digests[0], roll_lp_digests[0]

    # --- Rollout probe after alternating fillers ---
    for _ in range(args.alternate_rounds):
        _post_json(bu, "/rollout/generate", filler_rollout_body, t)
        _post_json(bu, "/v1/images/generations", filler_openai_body, t)
    rows = _post_json(bu, "/rollout/generate", probe_rollout_body, t)
    mixed_g, mixed_lp = _rollout_fingerprints(rows[0])
    if mixed_g != ref_roll_g or mixed_lp != ref_roll_lp:
        print("ROLLOUT PROBE changed after alternating requests.", file=sys.stderr)
        print(f"  generated_output sha256-json: ref={ref_roll_g} mixed={mixed_g}", file=sys.stderr)
        print(f"  rollout_log_probs sha256-json: ref={ref_roll_lp} mixed={mixed_lp}", file=sys.stderr)
        raise SystemExit(3)
    print("rollout probe after alternation: OK (matches isolated baseline)")

    # --- OpenAI probe: isolated baseline ---
    open_digests: list[str] = []
    for _ in range(args.baseline_repeats):
        resp = _post_json(bu, "/v1/images/generations", probe_openai_body, t)
        open_digests.append(_openai_image_fingerprint(resp))
    _assert_all_equal("openai probe image bytes", open_digests)
    ref_open = open_digests[0]

    # --- OpenAI probe after alternating fillers (rollout first to mirror the other test) ---
    for _ in range(args.alternate_rounds):
        _post_json(bu, "/rollout/generate", filler_rollout_body, t)
        _post_json(bu, "/v1/images/generations", filler_openai_body, t)
    resp = _post_json(bu, "/v1/images/generations", probe_openai_body, t)
    mixed_open = _openai_image_fingerprint(resp)
    if mixed_open != ref_open:
        print("OPENAI PROBE changed after alternating requests.", file=sys.stderr)
        print(f"  image sha256: ref={ref_open} mixed={mixed_open}", file=sys.stderr)
        raise SystemExit(4)
    print("openai probe after alternation: OK (matches isolated baseline)")

    print(
        "\nSummary: neither rollout nor /v1/images/generations probe output changed "
        f"after {args.alternate_rounds} rollout/openai filler rounds."
    )


if __name__ == "__main__":
    main()
