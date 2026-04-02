"""Rollout Image API — returns all denoising inputs, trajectory data, and rollout log-probs.

This endpoint is designed for RL post-training workflows. It reuses the
existing generation pipeline by constructing a standard ``Req`` with rollout
flags enabled, so zero changes are needed in the scheduler or denoising stages.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import ORJSONResponse

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutImageRequest,
    RolloutImageResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_serialize,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

router = APIRouter(prefix="/rollout", tags=["rollout"])


def _serialize_rollout_trajectory(
    rtd: RolloutTrajectoryData | None,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """Serialize rollout log-probs, debug tensors, static denoising env, and DiT trajectory."""
    if rtd is None:
        return None, None, None, None

    log_probs = _maybe_serialize(rtd.rollout_log_probs) if rtd.rollout_log_probs is not None else None

    debug_tensors = None
    if rtd.rollout_debug_tensors is not None:
        dt = rtd.rollout_debug_tensors
        debug_tensors = {
            "rollout_variance_noises": _maybe_serialize(dt.rollout_variance_noises),
            "rollout_prev_sample_means": _maybe_serialize(dt.rollout_prev_sample_means),
            "rollout_noise_std_devs": _maybe_serialize(dt.rollout_noise_std_devs),
            "rollout_model_outputs": _maybe_serialize(dt.rollout_model_outputs),
        }

    denoising_env = None
    if rtd.denoising_env is not None:
        env = rtd.denoising_env
        denoising_env = {
            "static": {
                "image_kwargs": _maybe_serialize(env.image_kwargs) if env.image_kwargs else None,
                "pos_cond_kwargs": _maybe_serialize(env.pos_cond_kwargs) if env.pos_cond_kwargs else None,
                "neg_cond_kwargs": _maybe_serialize(env.neg_cond_kwargs) if env.neg_cond_kwargs else None,
                "guidance": _maybe_serialize(env.guidance) if env.guidance is not None else None,
            },
        }

    dit_trajectory = None
    if rtd.dit_trajectory is not None:
        dtr = rtd.dit_trajectory
        dit_trajectory = {
            "latent_model_inputs": (
                _maybe_serialize(dtr.latent_model_inputs)
                if dtr.latent_model_inputs is not None
                else None
            ),
            "timesteps": (
                _maybe_serialize(dtr.timesteps) if dtr.timesteps is not None else None
            ),
        }

    return log_probs, debug_tensors, denoising_env, dit_trajectory


def _build_response(
    request_id: str,
    prompt: str,
    seed: int,
    result: OutputBatch,
) -> RolloutImageResponse:
    """Assemble the rollout response from an OutputBatch."""
    generated_output = _maybe_serialize(result.output) if result.output is not None else None

    trajectory_latents = _maybe_serialize(result.trajectory_latents) if result.trajectory_latents is not None else None
    trajectory_timesteps = _maybe_serialize(result.trajectory_timesteps) if result.trajectory_timesteps is not None else None

    rollout_log_probs, rollout_debug_tensors, denoising_env, dit_trajectory = (
        _serialize_rollout_trajectory(result.rollout_trajectory_data)
    )

    return RolloutImageResponse(
        request_id=request_id,
        prompt=prompt,
        seed=seed,
        generated_output=generated_output,
        trajectory_latents=trajectory_latents,
        trajectory_timesteps=trajectory_timesteps,
        rollout_log_probs=rollout_log_probs,
        rollout_debug_tensors=rollout_debug_tensors,
        denoising_env=denoising_env,
        dit_trajectory=dit_trajectory,
        inference_time_s=(
            result.metrics.total_duration_s
            if result.metrics and result.metrics.total_duration_s > 0
            else None
        ),
        peak_memory_mb=result.peak_memory_mb if result.peak_memory_mb > 0 else None,
    )


@router.post("/images", response_model=RolloutImageResponse)
async def rollout_images(request: RolloutImageRequest):
    """Generate an image with full rollout trajectory and log-prob data."""
    request_id = generate_request_id()
    server_args = get_global_server_args()

    sampling_kwargs: dict = dict(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        generator_device=request.generator_device,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        true_cfg_scale=request.true_cfg_scale,
        image_path=request.image_path,
        # force rollout flags
        rollout=True,
        rollout_sde_type=request.rollout_sde_type,
        rollout_noise_level=request.rollout_noise_level,
        rollout_log_prob_no_const=request.rollout_log_prob_no_const,
        rollout_debug_mode=request.rollout_debug_mode,
        return_trajectory_latents=request.return_trajectory_latents,
        return_trajectory_decoded=request.return_trajectory_decoded,
        return_dit_env=request.return_dit_env,
        # disable saving to disk — caller wants raw tensor data
        save_output=False,
    )

    if request.extra_sampling_params:
        sampling_kwargs.update(request.extra_sampling_params)

    # filter None values so SamplingParams defaults are used
    sampling_kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}

    try:
        sp = build_sampling_params(request_id, **sampling_kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid sampling params: {e}") from e

    req = prepare_request(server_args=server_args, sampling_params=sp)

    try:
        result: OutputBatch = await async_scheduler_client.forward(req)
    except Exception as e:
        logger.error("Rollout generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    response = _build_response(request_id, request.prompt, request.seed, result)
    return ORJSONResponse(content=response.model_dump())
