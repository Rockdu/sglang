"""Request/response data structures for post-training APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field


@dataclass
class UpdateWeightFromDiskReqInput:
    """Request to update model weights from disk for diffusion models."""

    model_path: str
    flush_cache: bool = True
    target_modules: list[str] | None = None


@dataclass
class GetWeightsChecksumReqInput:
    """Compute SHA-256 checksum of loaded module weights for verification."""

    module_names: list[str] | None = None


# ---------------------------------------------------------------------------
# Rollout Image API
# ---------------------------------------------------------------------------


class RolloutImageRequest(BaseModel):
    """Request body for ``POST /rollout/images``."""

    prompt: str
    negative_prompt: Optional[str] = None
    seed: int = 1024
    generator_device: str = "cuda"

    # geometry
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    num_outputs_per_prompt: Optional[int] = None

    # guidance
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = None

    # rollout-specific
    rollout_sde_type: str = "sde"
    rollout_noise_level: float = 0.7
    rollout_log_prob_no_const: bool = False
    rollout_debug_mode: bool = False

    # optional DiT capture (ODE/VAE per-step decode is not exposed on this endpoint)
    rollout_return_denoising_env: bool = False  # conditioning fields in ``denoising_env``
    rollout_return_dit_trajectory: bool = False  # per-step inputs in ``dit_trajectory``

    # image input (for I2I / TI2I tasks)
    image_path: Optional[list[str]] = None

    # pass-through for model-specific overrides
    extra_sampling_params: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional SamplingParams fields forwarded verbatim.",
    )


class RolloutImageResponse(BaseModel):
    """Response body for ``POST /rollout/images``."""

    request_id: str
    prompt: str
    seed: int

    generated_output: Any = None  # decoded image etc.; same structure as single-request, sliced to this sample

    rollout_log_probs: Optional[dict[str, Any]] = None
    rollout_debug_tensors: Optional[dict[str, Any]] = None  # present when ``rollout_debug_mode=True``
    denoising_env: Optional[dict[str, Any]] = None  # present when ``rollout_return_denoising_env=True``
    dit_trajectory: Optional[dict[str, Any]] = None  # present when ``rollout_return_dit_trajectory=True``

    inference_time_s: Optional[float] = None  # full forward duration; duplicated with the same value per sample
    peak_memory_mb: Optional[float] = None
