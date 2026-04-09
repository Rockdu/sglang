# Rollout `/rollout/generate` response shape report

**Generated:** 2026-04-06 05:38:44 UTC

> **Note:** This run used `--dry-run`: task types and the model list are filled in, but there are **no** per-model JSON shape outlines yet. On a machine with weights and GPU, run without `--dry-run` to record real `__tensor_shape__` trees.

Serialized tensors in API JSON use `__tensor__` with base64 `data`. This report uses the same shape outline convention as `test_rollout_print_response_shapes.py`: each tensor becomes an object with `__tensor_shape__` and optional `dtype`.

## Source of model list

Paths are the union of `sglang.multimodal_gen.registry._MODEL_HF_PATH_TO_NAME` keys and any `--models` you pass. Entries that only use `model_detectors` (no explicit HF id in the registry) are **not** auto-included.

## Run configuration

- **Dry run:** `True` (no HTTP, no server).
- **Include image-input models (I2V / I2I / I2M):** `False`.
- **Fixture image paths:** `None`.

## Summary

| Model | Task type | Status | Notes |
|-------|-----------|--------|-------|
| `BestWishYsh/Helios-Base` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `BestWishYsh/Helios-Distilled` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `BestWishYsh/Helios-Mid` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Efficient-Large-Model/Sana_1600M_1024px_diffusers` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Efficient-Large-Model/Sana_1600M_512px_diffusers` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Efficient-Large-Model/Sana_600M_1024px_diffusers` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Efficient-Large-Model/Sana_600M_512px_diffusers` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `FastVideo/FastHunyuan-diffusers` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `FastVideo/FastWan2.2-TI2V-5B-Diffusers` | TI2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | TI2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `FireRedTeam/FireRed-Image-Edit-1.0` | I2I | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `FireRedTeam/FireRed-Image-Edit-1.1` | I2I | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `IPostYellow/TurboWan2.1-T2V-14B-Diffusers` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers` | I2V | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `Qwen/Qwen-Image` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Qwen/Qwen-Image-2512` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Qwen/Qwen-Image-Edit` | I2I | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `Qwen/Qwen-Image-Edit-2509` | I2I | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `Qwen/Qwen-Image-Edit-2511` | I2I | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `Qwen/Qwen-Image-Layered` | I2I | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `Tongyi-MAI/Z-Image` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Tongyi-MAI/Z-Image-Turbo` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | I2V | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | I2V | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | I2V | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | TI2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `black-forest-labs/FLUX.1-dev` | T2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `black-forest-labs/FLUX.2-dev` | TI2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `black-forest-labs/FLUX.2-klein-4B` | TI2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `black-forest-labs/FLUX.2-klein-9B` | TI2I | SKIPPED | dry-run (no server started, no HTTP request) |
| `hunyuanvideo-community/HunyuanVideo` | T2V | SKIPPED | dry-run (no server started, no HTTP request) |
| `tencent/Hunyuan3D-2` | I2M | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |
| `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` | I2V | SKIPPED | Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image. |

## Per-model details

### `BestWishYsh/Helios-Base`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `BestWishYsh/Helios-Distilled`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `BestWishYsh/Helios-Mid`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Efficient-Large-Model/Sana_1600M_1024px_diffusers`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Efficient-Large-Model/Sana_1600M_512px_diffusers`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Efficient-Large-Model/Sana_600M_1024px_diffusers`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Efficient-Large-Model/Sana_600M_512px_diffusers`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `FastVideo/FastHunyuan-diffusers`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `FastVideo/FastWan2.2-TI2V-5B-Diffusers`

- **Task type:** TI2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`

- **Task type:** TI2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `FireRedTeam/FireRed-Image-Edit-1.0`

- **Task type:** I2I
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `FireRedTeam/FireRed-Image-Edit-1.1`

- **Task type:** I2I
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `IPostYellow/TurboWan2.1-T2V-14B-Diffusers`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers`

- **Task type:** I2V
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `Qwen/Qwen-Image`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Qwen/Qwen-Image-2512`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Qwen/Qwen-Image-Edit`

- **Task type:** I2I
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `Qwen/Qwen-Image-Edit-2509`

- **Task type:** I2I
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `Qwen/Qwen-Image-Edit-2511`

- **Task type:** I2I
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `Qwen/Qwen-Image-Layered`

- **Task type:** I2I
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `Tongyi-MAI/Z-Image`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Tongyi-MAI/Z-Image-Turbo`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`

- **Task type:** I2V
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`

- **Task type:** I2V
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `Wan-AI/Wan2.1-T2V-14B-Diffusers`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `Wan-AI/Wan2.2-I2V-A14B-Diffusers`

- **Task type:** I2V
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `Wan-AI/Wan2.2-TI2V-5B-Diffusers`

- **Task type:** TI2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `black-forest-labs/FLUX.1-dev`

- **Task type:** T2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `black-forest-labs/FLUX.2-dev`

- **Task type:** TI2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `black-forest-labs/FLUX.2-klein-4B`

- **Task type:** TI2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `black-forest-labs/FLUX.2-klein-9B`

- **Task type:** TI2I
- **Skipped:** dry-run (no server started, no HTTP request)

### `hunyuanvideo-community/HunyuanVideo`

- **Task type:** T2V
- **Skipped:** dry-run (no server started, no HTTP request)

### `tencent/Hunyuan3D-2`

- **Task type:** I2M
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

### `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers`

- **Task type:** I2V
- **Skipped:** Model task requires image input (I2V/I2I/I2M); re-run with --include-image-input-models and --fixture-image.

