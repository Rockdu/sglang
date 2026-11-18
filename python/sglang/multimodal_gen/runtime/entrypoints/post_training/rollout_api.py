"""Rollout HTTP API (``POST /rollout/generate``)."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import torch
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from sglang.multimodal_gen.configs.post_training.rl_rollout import (
    AUDIO_STREAM,
    PRIMARY_STREAM,
)
from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutRequest,
    RolloutResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.output_store import (
    rollout_output_store,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.request_timing import (
    STAGES_HEADER,
    TIMING_HEADER,
    RequestStamps,
    stages_header,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_serialize,
    _quantize_video_uint8,
    msgpack_encode_spliced,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
    select_output_audio,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    SKIP_DECODE_UNSUPPORTED_ERROR,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutStreamData,
    RolloutTrajectoryData,
    map_stream_tensors,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from radix_transfer_plugins import StoreRef

logger = init_logger(__name__)
router = APIRouter(prefix="/rollout", tags=["rollout"])


def _extract_single_sample_tensor(
    obj: Any, sample_idx: int, batch_size: int, *, current_key: str | None = None
) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 1 and obj.shape[0] == batch_size:
            return obj[sample_idx].contiguous()
        return obj
    if isinstance(obj, dict):
        return {
            k: _extract_single_sample_tensor(v, sample_idx, batch_size, current_key=k)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        if current_key == "img_shapes" and len(obj) == batch_size:
            return [obj[sample_idx]]
        return [
            _extract_single_sample_tensor(
                v, sample_idx, batch_size, current_key=current_key
            )
            for v in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            _extract_single_sample_tensor(
                v, sample_idx, batch_size, current_key=current_key
            )
            for v in obj
        )
    return obj


def _slice_stream_for_sample(
    stream: RolloutStreamData | None, sample_idx: int, batch_size: int
) -> RolloutStreamData | None:
    return map_stream_tensors(
        stream,
        lambda tensor: _extract_single_sample_tensor(tensor, sample_idx, batch_size),
    )


def _slice_rollout_trajectory_for_sample(
    rtd: RolloutTrajectoryData | None,
    sample_idx: int,
    batch_size: int,
) -> RolloutTrajectoryData | None:
    if rtd is None:
        return None
    denoising_env = None
    if rtd.denoising_env:
        env = rtd.denoising_env
        denoising_env = RolloutDenoisingEnv(
            image_kwargs=(
                _extract_single_sample_tensor(env.image_kwargs, sample_idx, batch_size)
                if env.image_kwargs
                else None
            ),
            pos_cond_kwargs=(
                _extract_single_sample_tensor(
                    env.pos_cond_kwargs, sample_idx, batch_size
                )
                if env.pos_cond_kwargs
                else None
            ),
            neg_cond_kwargs=(
                _extract_single_sample_tensor(
                    env.neg_cond_kwargs, sample_idx, batch_size
                )
                if env.neg_cond_kwargs
                else None
            ),
            guidance=(
                _extract_single_sample_tensor(env.guidance, sample_idx, batch_size)
                if env.guidance is not None
                else None
            ),
        )
    return RolloutTrajectoryData(
        denoising_env=denoising_env,
        streams={
            name: _slice_stream_for_sample(stream, sample_idx, batch_size)
            for name, stream in rtd.streams.items()
        },
    )


def _store_leaves(
    obj: Any,
    sample_idx: int,
    batch_size: int,
    *,
    path: str,
    stored: dict[str, torch.Tensor],
    current_key: str | None = None,
) -> Any:
    """``_extract_single_sample_tensor`` with every tensor put in ``stored`` and a marker in its place."""
    if isinstance(obj, torch.Tensor):
        # One field per sample of a batched tensor, so a reader fetches only its samples.
        if obj.dim() >= 1 and obj.shape[0] == batch_size:
            path, obj = f"{path}.{sample_idx}", obj[sample_idx]
        stored[path] = obj
        return {"__store__": path}
    if isinstance(obj, dict):
        return {
            k: _store_leaves(
                v,
                sample_idx,
                batch_size,
                path=f"{path}.{k}",
                stored=stored,
                current_key=k,
            )
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        if current_key == "img_shapes" and len(obj) == batch_size:
            return [obj[sample_idx]]
        return [
            _store_leaves(
                v,
                sample_idx,
                batch_size,
                path=f"{path}.{i}",
                stored=stored,
                current_key=current_key,
            )
            for i, v in enumerate(obj)
        ]
    return _maybe_serialize(obj)


def _store_steps(
    tensor: torch.Tensor | None,
    sample_idx: int,
    *,
    path: str,
    stored: dict[str, torch.Tensor],
    steps: list[int] | None = None,
) -> list[dict] | None:
    """One field per step of a ``[B, steps, ...]`` tensor's sample, named by its original step number."""
    if tensor is None:
        return None
    if steps is None:
        steps = range(tensor.shape[1])
    markers = []
    for position, step in enumerate(steps):
        field = f"{path}.{sample_idx}.{step}"
        stored[field] = tensor[sample_idx, position]
        markers.append({"__store__": field})
    return markers


def _serialize_stream_via_store(
    stream: RolloutStreamData,
    sample_idx: int,
    batch_size: int,
    *,
    path_prefix: str,
    stored: dict[str, torch.Tensor],
    serialized_dit_timesteps: dict | None,
    serialized_dit_sigmas: dict | None,
) -> dict:
    log_probs = stream.rollout_log_probs
    if (
        isinstance(log_probs, torch.Tensor)
        and log_probs.dim() >= 1
        and log_probs.shape[0] == batch_size
    ):
        log_probs = log_probs[sample_idx].contiguous()
    serialized_debug_tensors = None
    if stream.rollout_debug_tensors:
        serialized_debug_tensors = {
            name: _store_steps(
                tensor,
                sample_idx,
                path=f"{path_prefix}rollout_debug_tensors.{name}",
                stored=stored,
            )
            for name, tensor in vars(stream.rollout_debug_tensors).items()
        }
    serialized_dit_trajectory = None
    if stream.dit_trajectory:
        dit = stream.dit_trajectory
        serialized_dit_trajectory = {
            "latents": _store_steps(
                dit.latents,
                sample_idx,
                path=f"{path_prefix}dit_trajectory.latents",
                stored=stored,
                steps=(
                    dit.latent_step_indices.tolist()
                    if dit.latent_step_indices is not None
                    else None
                ),
            ),
            "timesteps": serialized_dit_timesteps,
            "sigmas": serialized_dit_sigmas,
            "latent_step_indices": _maybe_serialize(dit.latent_step_indices),
        }
    return {
        "rollout_log_probs": _maybe_serialize(log_probs),
        "rollout_debug_tensors": serialized_debug_tensors,
        "dit_trajectory": serialized_dit_trajectory,
    }


def _serialize_rollout_trajectory_via_store(
    rtd: RolloutTrajectoryData,
    sample_idx: int,
    batch_size: int,
    *,
    stored: dict[str, torch.Tensor],
    serialized_dit_timesteps: dict | None,
    serialized_dit_sigmas: dict | None,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """``_serialize_rollout_trajectory`` of the batched trajectory, its large tensors going to ``stored``."""
    serialized_stream = _serialize_stream_via_store(
        rtd.streams[PRIMARY_STREAM],
        sample_idx,
        batch_size,
        path_prefix="",
        stored=stored,
        serialized_dit_timesteps=serialized_dit_timesteps,
        serialized_dit_sigmas=serialized_dit_sigmas,
    )
    serialized_denoising_env = None
    if rtd.denoising_env:
        env = rtd.denoising_env
        serialized_denoising_env = {
            name: (
                _store_leaves(
                    kwargs,
                    sample_idx,
                    batch_size,
                    path=f"denoising_env.{name}",
                    stored=stored,
                )
                if kwargs
                else None
            )
            for name, kwargs in (
                ("image_kwargs", env.image_kwargs),
                ("pos_cond_kwargs", env.pos_cond_kwargs),
                ("neg_cond_kwargs", env.neg_cond_kwargs),
            )
        }
        serialized_denoising_env["guidance"] = (
            _maybe_serialize(
                _extract_single_sample_tensor(env.guidance, sample_idx, batch_size)
            )
            if env.guidance is not None
            else None
        )
    return (
        serialized_stream["rollout_log_probs"],
        serialized_stream["rollout_debug_tensors"],
        serialized_denoising_env,
        serialized_stream["dit_trajectory"],
    )


def _serialize_stream(
    stream: RolloutStreamData | None,
    *,
    serialized_dit_timesteps: dict | None = None,
    serialized_dit_sigmas: dict | None = None,
) -> dict:
    debug = stream.rollout_debug_tensors if stream else None
    trajectory = stream.dit_trajectory if stream else None
    return {
        "rollout_log_probs": _maybe_serialize(stream.rollout_log_probs)
        if stream
        else None,
        "rollout_debug_tensors": {
            field.name: _maybe_serialize(getattr(debug, field.name))
            for field in fields(debug)
        }
        if debug is not None
        else None,
        "dit_trajectory": {
            "latents": _maybe_serialize(trajectory.latents),
            "timesteps": serialized_dit_timesteps,
            "sigmas": serialized_dit_sigmas,
            "latent_step_indices": _maybe_serialize(trajectory.latent_step_indices),
        }
        if trajectory is not None
        else None,
    }


def _serialize_rollout_trajectory(
    rtd: RolloutTrajectoryData | None,
    *,
    serialized_dit_timesteps: dict | None = None,
    serialized_dit_sigmas: dict | None = None,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """Return order: rollout_log_probs, rollout_debug_tensors, denoising_env, dit_trajectory."""
    serialized_stream = _serialize_stream(
        rtd.streams[PRIMARY_STREAM] if rtd is not None else None,
        serialized_dit_timesteps=serialized_dit_timesteps,
        serialized_dit_sigmas=serialized_dit_sigmas,
    )
    serialized_denoising_env = None
    if rtd is not None and rtd.denoising_env:
        env = rtd.denoising_env
        serialized_denoising_env = {
            "image_kwargs": (
                _maybe_serialize(env.image_kwargs) if env.image_kwargs else None
            ),
            "pos_cond_kwargs": (
                _maybe_serialize(env.pos_cond_kwargs) if env.pos_cond_kwargs else None
            ),
            "neg_cond_kwargs": (
                _maybe_serialize(env.neg_cond_kwargs) if env.neg_cond_kwargs else None
            ),
            "guidance": (
                _maybe_serialize(env.guidance) if env.guidance is not None else None
            ),
        }
    return (
        serialized_stream["rollout_log_probs"],
        serialized_stream["rollout_debug_tensors"],
        serialized_denoising_env,
        serialized_stream["dit_trajectory"],
    )


def _serialize_schedule(
    dit_trajectory: RolloutDitTrajectory | None,
) -> tuple[dict | None, dict | None]:
    if dit_trajectory is None:
        return None, None
    return (
        _maybe_serialize(dit_trajectory.timesteps),
        _maybe_serialize(dit_trajectory.sigmas),
    )


def _serialize_sample_output(
    value: Any, *, store_field: str, stored: dict[str, torch.Tensor] | None
) -> Any:
    if stored is None or value is None:
        return _maybe_serialize(value)
    # some pipelines return a numpy array, which the inline path serializes as a tensor too
    stored[store_field] = torch.as_tensor(value)
    return {"__store__": store_field}


def _extra_stream_outputs(
    result: OutputBatch, sample_idx: int, batch_size: int
) -> dict[str, dict[str, Any]]:
    audio = result.audio
    if audio is None:
        return {}
    # A 1-D track is shared by every sample (see ``select_output_audio``).
    if audio.ndim >= 2 and audio.shape[0] != batch_size:
        raise ValueError(
            f"audio has {audio.shape[0]} outputs for a batch of {batch_size}"
        )
    return {
        AUDIO_STREAM: {
            "generated_output": select_output_audio(audio, sample_idx),
            "sample_rate": result.audio_sample_rate,
        }
    }


def _serialize_extra_streams(
    outputs: dict[str, dict[str, Any]],
    extra_stream_rollouts: dict[str, RolloutStreamData],
    sample_idx: int,
    batch_size: int,
    *,
    stored: dict[str, torch.Tensor] | None,
    schedules: dict[str, tuple[dict | None, dict | None]],
) -> dict[str, dict] | None:
    serialized = {}
    for name in dict.fromkeys((*outputs, *extra_stream_rollouts)):
        stream = extra_stream_rollouts.get(name)
        timesteps, sigmas = schedules.get(name, (None, None))
        if stream is None or stored is None:
            rollout_fields = _serialize_stream(
                _slice_stream_for_sample(stream, sample_idx, batch_size),
                serialized_dit_timesteps=timesteps,
                serialized_dit_sigmas=sigmas,
            )
        else:
            rollout_fields = _serialize_stream_via_store(
                stream,
                sample_idx,
                batch_size,
                path_prefix=f"extra_streams.{name}.",
                stored=stored,
                serialized_dit_timesteps=timesteps,
                serialized_dit_sigmas=sigmas,
            )
        output = dict(outputs.get(name, {}))
        serialized[name] = {
            "generated_output": _serialize_sample_output(
                output.pop("generated_output", None),
                store_field=f"extra_streams.{name}.generated_output.{sample_idx}",
                stored=stored,
            ),
            **output,
            **rollout_fields,
        }
    return serialized or None


def _build_response(
    request_id: str,
    prompt: str,
    seed: int,
    rollout: bool,
    result: OutputBatch,
    video_dtype: str | None = None,
    stored: dict[str, torch.Tensor] | None = None,
    num_outputs: int | None = None,
) -> list[RolloutResponse]:
    """
    rollout: bool - set to False when evaluating the model
    stored: collects the batched tensors to put in the output store, or None to inline them
    num_outputs: the response count when the batch carries no decoded output
    """
    batch_size = result.output.shape[0] if result.output is not None else num_outputs
    inference_time_s = (
        result.metrics.total_duration_s
        if result.metrics and result.metrics.total_duration_s > 0
        else None
    )
    peak_memory_mb = result.peak_memory_mb if result.peak_memory_mb > 0 else None
    rollout_trajectory_data = result.rollout_trajectory_data
    primary = None
    extra_stream_rollouts = {}
    if rollout:
        assert rollout_trajectory_data is not None, (
            "rollout_trajectory_data must be present when rollout=True"
        )
        primary = rollout_trajectory_data.streams[PRIMARY_STREAM]
        extra_stream_rollouts = {
            name: stream
            for name, stream in rollout_trajectory_data.streams.items()
            if name != PRIMARY_STREAM
        }
    serialized_dit_timesteps, serialized_dit_sigmas = _serialize_schedule(
        primary.dit_trajectory if primary is not None else None
    )
    extra_stream_schedules = {
        name: _serialize_schedule(stream.dit_trajectory)
        for name, stream in extra_stream_rollouts.items()
    }

    responses: list[RolloutResponse] = []
    for sample_idx in range(batch_size):
        out_i = result.output[sample_idx] if result.output is not None else None
        if isinstance(out_i, torch.Tensor):
            if video_dtype == "uint8":
                out_i = _quantize_video_uint8(out_i)
            out_i = out_i.contiguous()
        extra_stream_outputs = _extra_stream_outputs(result, sample_idx, batch_size)
        if not rollout:
            responses.append(
                RolloutResponse(
                    request_id=request_id,
                    prompt=prompt,
                    seed=seed,
                    generated_output=_maybe_serialize(out_i),
                    extra_streams=_serialize_extra_streams(
                        extra_stream_outputs,
                        {},
                        sample_idx,
                        batch_size,
                        stored=None,
                        schedules={},
                    ),
                    inference_time_s=inference_time_s,
                    peak_memory_mb=peak_memory_mb,
                )
            )
            continue
        if stored is None:
            serialized = _serialize_rollout_trajectory(
                _slice_rollout_trajectory_for_sample(
                    rollout_trajectory_data, sample_idx, batch_size
                ),
                serialized_dit_timesteps=serialized_dit_timesteps,
                serialized_dit_sigmas=serialized_dit_sigmas,
            )
        else:
            serialized = _serialize_rollout_trajectory_via_store(
                rollout_trajectory_data,
                sample_idx,
                batch_size,
                stored=stored,
                serialized_dit_timesteps=serialized_dit_timesteps,
                serialized_dit_sigmas=serialized_dit_sigmas,
            )
        (
            serialized_log_probs,
            serialized_debug_tensors,
            serialized_denoising_env,
            serialized_dit_trajectory,
        ) = serialized
        responses.append(
            RolloutResponse(
                request_id=request_id,
                prompt=prompt,
                seed=seed,
                generated_output=_serialize_sample_output(
                    out_i, store_field=f"generated_output.{sample_idx}", stored=stored
                ),
                rollout_log_probs=serialized_log_probs,
                rollout_debug_tensors=serialized_debug_tensors,
                denoising_env=serialized_denoising_env,
                dit_trajectory=serialized_dit_trajectory,
                extra_streams=_serialize_extra_streams(
                    extra_stream_outputs,
                    extra_stream_rollouts,
                    sample_idx,
                    batch_size,
                    stored=stored,
                    schedules=extra_stream_schedules,
                ),
                inference_time_s=inference_time_s,
                peak_memory_mb=peak_memory_mb,
            )
        )
    return responses


def _engine_error(error: str) -> HTTPException:
    status_code = 400 if error == SKIP_DECODE_UNSUPPORTED_ERROR else 500
    return HTTPException(status_code=status_code, detail=error)


def _build_sampling_kwargs(request: RolloutRequest) -> dict:
    sampling_kwargs: dict = dict(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        generator_device=request.generator_device,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        num_outputs_per_prompt=request.num_outputs_per_prompt,
        guidance_scale=request.guidance_scale,
        true_cfg_scale=request.true_cfg_scale,
        num_frames=request.num_frames,
        fps=request.fps,
        image_path=request.image_path,
        rollout=request.rollout,
        rollout_extra_streams=tuple(request.rollout_extra_streams),
        rollout_sde_type=request.rollout_sde_type,
        rollout_noise_level=request.rollout_noise_level,
        rollout_log_prob_no_const=request.rollout_log_prob_no_const,
        rollout_debug_mode=request.rollout_debug_mode,
        rollout_return_denoising_env=request.rollout_return_denoising_env,
        rollout_return_dit_trajectory=request.rollout_return_dit_trajectory,
        rollout_return_generated_output=request.rollout_return_generated_output,
        rollout_sde_step_indices=request.rollout_sde_step_indices,
        rollout_return_step_indices=request.rollout_return_step_indices,
        rollout_stop_step_index=request.rollout_stop_step_index,
        suppress_logs=request.suppress_logs,
        save_output=False,
        return_trajectory_latents=False,
        return_trajectory_decoded=False,
    )
    if request.extra_sampling_params:
        sampling_kwargs.update(request.extra_sampling_params)
        sampling_kwargs["rollout"] = request.rollout
    return {k: v for k, v in sampling_kwargs.items() if v is not None}


def _stream_parts(parts: list[bytes], ref: StoreRef | None):
    sent = False
    try:
        yield from parts
        sent = True
    finally:
        if ref is not None and not sent:
            rollout_output_store.delete(ref)


@router.post(
    "/generate",
    response_class=Response,
    responses={
        200: {
            "model": list[RolloutResponse],
            "content": {"application/msgpack": {}},
        }
    },
)
async def rollout_generate(request: RolloutRequest):
    stamps = RequestStamps()
    stamps.mark("srv_recv")
    request_id = generate_request_id()
    stamps.request_id = request_id
    server_args = get_global_server_args()
    if request.rollout_return_outputs_via_store and not rollout_output_store.enabled:
        raise HTTPException(
            status_code=400,
            detail="rollout_return_outputs_via_store needs a server started with "
            "--output-store-backend",
        )
    sampling_kwargs = _build_sampling_kwargs(request)
    try:
        sampling_params = build_sampling_params(request_id, **sampling_kwargs)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid sampling params: {exc}"
        ) from exc
    try:
        pipeline_request = prepare_request(
            server_args=server_args, sampling_params=sampling_params
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request: {exc}") from exc
    try:
        with stamps.span("forward"):
            output_batch: OutputBatch = await async_scheduler_client.forward(
                pipeline_request
            )
    except Exception as exc:
        logger.error("Rollout generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc
    if output_batch.error:
        raise _engine_error(output_batch.error)

    def _serialize_response() -> tuple[list[bytes], StoreRef | None]:
        stored = {} if request.rollout_return_outputs_via_store else None
        with stamps.span("build"):
            rollout_responses = _build_response(
                request_id,
                request.prompt,
                request.seed,
                request.rollout,
                output_batch,
                video_dtype=request.rollout_video_dtype,
                stored=stored,
                num_outputs=sampling_params.num_outputs_per_prompt,
            )
        ref = None
        if stored:
            with stamps.span("store"):
                ref = rollout_output_store.put(stored)
            for rollout_response in rollout_responses:
                rollout_response.output_store_ref = ref.to_builtins()
        try:
            with stamps.span("dump"):
                payload = [r.model_dump() for r in rollout_responses]
            with stamps.span("msgpack"):
                return msgpack_encode_spliced(payload), ref
        except BaseException:
            if ref is not None:
                rollout_output_store.delete(ref)
            raise

    # Off the event loop: building the response copies hundreds of MB
    parts, ref = await asyncio.to_thread(_serialize_response)

    # Timing rides a header: the last marks postdate the encoded body. The
    # explicit content-length keeps uvicorn on identity framing, not chunked.
    headers = {
        TIMING_HEADER: stamps.to_header(),
        "content-length": str(sum(len(p) for p in parts)),
    }
    stages = stages_header(output_batch.metrics)
    if stages:
        headers[STAGES_HEADER] = stages
    return StreamingResponse(
        _stream_parts(parts, ref),
        media_type="application/msgpack",
        headers=headers,
    )
