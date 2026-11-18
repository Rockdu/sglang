# SPDX-License-Identifier: Apache-2.0
"""Rollout hooks for denoise loops.

A loop runs inside ``rollout_scope``, calls ``prepare_rollout_if_enabled`` for each
of its latent streams and then ``set_rollout_early_stop_if_enabled`` before SP
sharding and before reading ``batch.timesteps``, calls
``init_rollout_env_if_enabled`` at the loop start and
``finalize_rollout_if_enabled`` at the loop end, and passes
``**rollout_step_kwargs(batch, stream=...)`` to each stream's ``scheduler.step``.
Every hook is a no-op for serving requests.
"""

from __future__ import annotations

import functools
import hashlib
from contextlib import contextmanager
from typing import Any, Callable

import torch

from sglang.multimodal_gen.configs.post_training.rl_rollout import PRIMARY_STREAM
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.flow_rollout_controller import (
    FlowRolloutController,
    truncate_rollout_sigmas,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutStreamData,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_latents_for_sp,
)


def _drop_rollout_working_state(req: Req) -> None:
    req.rollout_controllers = {}
    req._rollout_denoising_env_state = None


@contextmanager
def rollout_scope(batch: Req):
    """Reset rollout state on entry; drop the controllers and env snapshot on exit."""
    if not batch.rollout:
        yield
        return
    _drop_rollout_working_state(batch)
    batch.rollout_trajectory_data = None
    try:
        yield
    except BaseException:
        batch.rollout_trajectory_data = None
        raise
    finally:
        _drop_rollout_working_state(batch)


def _kwargs_to_cpu(d: Any) -> Any:
    if isinstance(d, torch.Tensor):
        return d.detach().cpu()
    if isinstance(d, dict):
        return {k: _kwargs_to_cpu(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_kwargs_to_cpu(v) for v in d]
    if isinstance(d, tuple):
        return tuple(_kwargs_to_cpu(v) for v in d)
    return d


def rollout_step_kwargs(batch: Req, *, stream: str) -> dict[str, Any]:
    """The rollout-only keyword arguments of ``scheduler.step`` for this request:
    the selected stream controller's ``transition``. Empty for a serving request, so
    its step call is unchanged and works with any scheduler."""
    current_rollout_controller = batch.rollout_controllers.get(stream)
    if current_rollout_controller is None:
        return {}
    return {
        "transition": functools.partial(
            current_rollout_controller.transition, batch=batch
        )
    }


def _rolled_out_streams(batch: Req) -> tuple[str, ...]:
    return (PRIMARY_STREAM, *batch.rollout_extra_streams)


def loop_latent_sp_fns(pipeline_config: Any) -> tuple[Callable, Callable]:
    """``(shard_for_sp, gather_for_sp)`` of latents the pipeline config shards for SP."""

    def shard_for_sp(batch: Req, latents: torch.Tensor) -> torch.Tensor:
        return pipeline_config.shard_latents_for_sp(batch=batch, latents=latents)[0]

    def gather_for_sp(batch: Req, stacked_latents: torch.Tensor) -> torch.Tensor:
        return gather_stacked_latents_for_sp(
            pipeline_config=pipeline_config,
            batch=batch,
            stacked_latents=stacked_latents,
        )

    return shard_for_sp, gather_for_sp


def derive_stream_generators(
    generators: torch.Generator | list[torch.Generator] | None, *, stream: str
) -> torch.Generator | list[torch.Generator]:
    """RNGs from the request seed and stream name only, so every rank agrees."""
    if generators is None:
        raise ValueError(f"Rollout of stream {stream!r} requires request generators")

    def derive(generator: torch.Generator) -> torch.Generator:
        digest = hashlib.sha256(
            f"sglang/rollout/{stream}/{generator.initial_seed()}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        return torch.Generator(device=generator.device).manual_seed(seed)

    if isinstance(generators, torch.Generator):
        return derive(generators)
    return [derive(generator) for generator in generators]


def _stream_generators(
    batch: Req, stream: str
) -> torch.Generator | list[torch.Generator]:
    # The primary stream draws from the request's own RNGs, whatever else is rolled out.
    if stream == PRIMARY_STREAM:
        return batch.generator
    return derive_stream_generators(batch.generator, stream=stream)


def prepare_rollout_if_enabled(
    *,
    batch: Req,
    stream: str,
    scheduler: Any,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    shard_for_sp: Callable[[Any, torch.Tensor], torch.Tensor],
    gather_for_sp: Callable[[Any, torch.Tensor], torch.Tensor],
) -> None:
    """Prepare one latent stream of the loop: its controller if the request rolls
    it out, else its scheduler's grid cut at the stop step.

    Placement: before SP sharding, with ``latents`` full-shape and ``scheduler`` on
    its uncut grid; ``timesteps`` is what the loop iterates for this stream. A
    rolled-out stream steps along its controller's schedule, its scheduler's grid
    is not cut.
    """
    if not batch.rollout:
        return
    if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError(
            f"Rollout of stream {stream!r} needs a FlowMatch Euler scheduler, "
            f"got {type(scheduler).__name__}"
        )
    for other, controller in batch.rollout_controllers.items():
        if controller.uncut_num_steps != len(timesteps):
            raise ValueError(
                f"Rollout stream {stream!r} has {len(timesteps)} steps but stream "
                f"{other!r} has {controller.uncut_num_steps}; streams step together"
            )
    stop_step_index = batch.rollout_stop_step_index
    if stream not in _rolled_out_streams(batch):
        # Without a controller the stream keeps its scheduler's own Euler step.
        if stop_step_index is not None:
            scheduler.sigmas = truncate_rollout_sigmas(
                scheduler.sigmas, stop_step_index=stop_step_index
            )
            scheduler.timesteps = scheduler.timesteps[: stop_step_index + 1]
        return
    if stream in batch.rollout_controllers:
        raise RuntimeError(f"Rollout stream {stream!r} was already prepared")
    current_rollout_controller = FlowRolloutController(
        sde_type=batch.rollout_sde_type,
        noise_level=float(batch.rollout_noise_level),
        log_prob_no_const=batch.rollout_log_prob_no_const,
        debug_mode=batch.rollout_debug_mode,
        sde_step_indices=batch.rollout_sde_step_indices,
        generator=_stream_generators(batch, stream),
        return_dit_trajectory=batch.rollout_return_dit_trajectory,
        return_step_indices=batch.rollout_return_step_indices,
        shard_for_sp=shard_for_sp,
        gather_for_sp=gather_for_sp,
        sigmas=scheduler.sigmas,
        timesteps=timesteps,
        latents_shape=tuple(latents.shape),
    )
    # After the build: sigma_max comes from the uncut grid.
    if stop_step_index is not None:
        current_rollout_controller.stop_at(stop_step_index)
    batch.rollout_controllers = {
        **batch.rollout_controllers,
        stream: current_rollout_controller,
    }


def set_rollout_early_stop_if_enabled(*, batch: Req) -> None:
    """Make ``batch.timesteps`` the primary controller's schedule, which
    ``prepare_rollout_if_enabled`` cut to steps 0..``rollout_stop_step_index``.
    No-op without a rollout or without a stop.

    Placement: directly after the primary stream's prepare and before the loop
    reads ``batch.timesteps`` or derives anything from it: the step count for
    warmup, cache-dit or the progress bar, a second stream's scheduler clone
    (see the LTX-2 audio scheduler). Anything read earlier sees the full grid.
    """
    if not batch.rollout or batch.rollout_stop_step_index is None:
        return
    batch.timesteps = batch.rollout_controllers[PRIMARY_STREAM].timesteps


def init_rollout_env_if_enabled(
    *,
    batch: Req,
    image_kwargs: dict[str, Any],
    pos_cond_kwargs: dict[str, Any],
    neg_cond_kwargs: dict[str, Any],
    guidance: torch.Tensor | None,
) -> None:
    """Snapshot the DiT conditioning that the trainer replays."""
    if not batch.rollout:
        return
    collect_env = batch.rollout_return_denoising_env
    collect_traj = batch.rollout_return_dit_trajectory
    if not (collect_env or collect_traj):
        batch._rollout_denoising_env_state = None
        return

    if collect_env:
        env = RolloutDenoisingEnv(
            image_kwargs=_kwargs_to_cpu(image_kwargs),
            pos_cond_kwargs=_kwargs_to_cpu(pos_cond_kwargs),
            neg_cond_kwargs=(
                _kwargs_to_cpu(neg_cond_kwargs) if neg_cond_kwargs else None
            ),
            guidance=guidance.detach().cpu() if guidance is not None else None,
        )
        pos_src = pos_cond_kwargs
        neg_src = neg_cond_kwargs
    else:
        env = None
        pos_src = None
        neg_src = None

    batch._rollout_denoising_env_state = {
        "env": env,
        "pos_cond_kwargs_src": pos_src,
        "neg_cond_kwargs_src": neg_src,
    }


def finalize_rollout_if_enabled(
    *,
    batch: Req,
    latents: dict[str, torch.Tensor],
    num_inference_steps: int,
    final_timestep: torch.Tensor,
    pipeline_config: Any,
) -> None:
    """Finalize rollout-only outputs.

    Must be called before ``_post_denoising_loop`` so that each stream's
    ``latents`` (its last ``scheduler.step`` output) is still SP-sharded and can
    be gathered uniformly with the per-step trajectory latents.
    """
    if not batch.rollout:
        return
    missing = batch.rollout_controllers.keys() - latents.keys()
    if missing:
        raise RuntimeError(f"No final latents for rollout streams {sorted(missing)}")
    unprepared = set(_rolled_out_streams(batch)) - batch.rollout_controllers.keys()
    if unprepared:
        raise RuntimeError(
            f"Requested rollout streams {sorted(unprepared)} were not prepared by "
            "the denoising stage"
        )
    # Publish only once everything is collected, so a failure leaves no partial trajectory.
    streams = {
        stream: _collect_stream_rollout(
            batch=batch,
            controller=controller,
            latents=latents[stream],
            num_inference_steps=num_inference_steps,
            final_timestep=final_timestep,
        )
        for stream, controller in batch.rollout_controllers.items()
    }
    env = None
    state = batch._rollout_denoising_env_state
    if state is not None and state["env"] is not None:
        env: RolloutDenoisingEnv = state["env"]
        env.pos_cond_kwargs = _kwargs_to_cpu(
            pipeline_config.gather_denoising_env_static_for_sp(
                batch, state["pos_cond_kwargs_src"]
            )
        )
        if env.neg_cond_kwargs is not None:
            env.neg_cond_kwargs = _kwargs_to_cpu(
                pipeline_config.gather_denoising_env_static_for_sp(
                    batch, state["neg_cond_kwargs_src"]
                )
            )
    batch.rollout_trajectory_data = RolloutTrajectoryData(
        denoising_env=env, streams=streams
    )


def _collect_stream_rollout(
    *,
    batch: Req,
    controller: FlowRolloutController,
    latents: torch.Tensor,
    num_inference_steps: int,
    final_timestep: torch.Tensor,
) -> RolloutStreamData:
    num_schedule_steps = len(controller.timesteps)
    if num_inference_steps != num_schedule_steps:
        raise ValueError(
            f"the denoising loop ran {num_inference_steps} steps but the rollout "
            f"schedule has {num_schedule_steps}; a rollout loop must iterate the "
            "controller's timesteps"
        )
    data = RolloutStreamData(rollout_log_probs=controller.collect_rollout_log_probs())
    if controller.debug_mode:
        data.rollout_debug_tensors = controller.collect_rollout_debug_tensors(batch)
    controller.record_trajectory_step(
        latents=latents,
        timestep_value=final_timestep,
        step_index=num_inference_steps,
    )
    if controller.step_latents:
        data.dit_trajectory = RolloutDitTrajectory(
            latents=controller.gather_for_sp(
                batch, torch.stack(controller.step_latents, dim=1)
            ).cpu(),
            timesteps=torch.stack(controller.step_timesteps).cpu(),
            sigmas=controller.sigmas.detach().cpu().clone(),
            latent_step_indices=torch.tensor(
                controller.step_latent_indices, dtype=torch.long
            ),
        )
    return data
