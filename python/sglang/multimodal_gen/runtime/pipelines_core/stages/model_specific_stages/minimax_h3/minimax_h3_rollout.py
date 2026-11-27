# SPDX-License-Identifier: Apache-2.0
"""Rollout (RL) glue for the MiniMax H3 denoise loop.

H3 has no scheduler: its loop updates packed video and audio target rows in
place. Each stream still gets a shared ``FlowRolloutController``; this module
adapts the H3 rows to it and builds the denoising env.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import msgspec
import torch

from sglang.multimodal_gen.configs.post_training.rl_rollout import (
    AUDIO_STREAM,
    PRIMARY_STREAM,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_ROW_WIDTH,
    MINIMAX_H3_VIDEO_ROW_WIDTH,
    MiniMaxH3RolloutStep,
    MiniMaxH3StepTensors,
    _minimax_h3_step_tensors,
    _minimax_h3_update_target_rows_,
)
from sglang.multimodal_gen.runtime.post_training.flow_rollout_controller import (
    truncate_rollout_sigmas,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import (
    finalize_rollout_if_enabled,
    pipeline_config_sp_fns,
    prepare_rollout_if_enabled,
    publish_rollout_if_enabled,
    rollout_step_kwargs,
)


class MiniMaxH3RolloutPlan(msgspec.Struct, frozen=True):
    """The sigma grids a request's loop runs, cut at the rollout stop step, and
    each stream's rollout step (None: that stream takes the serving update)."""

    sigmas_video: list[float]
    sigmas_audio: list[float]
    video_rollout_step: MiniMaxH3RolloutStep | None = None
    audio_rollout_step: MiniMaxH3RolloutStep | None = None


def _minimax_h3_float_step_tensors(
    sigmas: list[float], *, device: torch.device
) -> MiniMaxH3StepTensors:
    """The operands of H3's video rollout ODE step: sigma and the sigma ratio
    rounded from Python doubles, a few ulp from ``_minimax_h3_step_tensors``.

    Kept so H3 video rollouts stay bit-identical to the trajectories the
    miles_diffusion H3 e2e standards were recorded on.
    TODO(rockdu): use _minimax_h3_step_tensors once those standards are re-recorded.
    """
    ratios = [
        sigma_next / sigma if sigma != 0.0 else 1.0
        for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:])
    ]
    sigma_ratios = torch.tensor(ratios, dtype=torch.float32, device=device)
    return MiniMaxH3StepTensors(
        sigma_t=torch.tensor(sigmas[:-1], dtype=torch.float32, device=device),
        sigma_ratios=sigma_ratios,
        one_minus_sigma_ratios=1.0 - sigma_ratios,
    )


def _minimax_h3_ode_update(
    *,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    step_index: int,
    sigmas: list[float],
    step_tensors: MiniMaxH3StepTensors,
) -> torch.Tensor:
    """The H3 loop's target-row update on a copy of ``sample``, as a controller's
    ``ode_transition``; ``model_output`` is the flow-matching ``-v``."""
    out = sample[0].clone()
    _minimax_h3_update_target_rows_(
        out,
        -model_output[0],
        sigma_t=step_tensors.sigma_t[step_index],
        sigma_curr=sigmas[step_index],
        sigma_ratio=step_tensors.sigma_ratios[step_index],
        one_minus_sigma_ratio=step_tensors.one_minus_sigma_ratios[step_index],
        denoised_scratch=torch.empty_like(out),
    )
    return out[None]


def _rollout_step(
    *,
    transition: Callable[..., torch.Tensor],
    sigmas: list[float],
    replay_timesteps: torch.Tensor,
    device: torch.device,
) -> MiniMaxH3RolloutStep:
    """Adapt one stream's ``[n, C]`` target rows to its controller's transition.

    The controller takes the flow-matching ``model_output = -v`` and records the
    sample it is given without a copy, while H3 updates its rows in place.
    """
    sigmas_t = torch.tensor(sigmas, dtype=torch.float32, device=device)

    def step(rows: torch.Tensor, velocity: torch.Tensor, step_index: int) -> None:
        current_sigma = sigmas_t[step_index]
        next_sigma = sigmas_t[step_index + 1]
        prev_sample = transition(
            model_output=(-velocity)[None],
            sample=rows[None].clone(),
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            dt=next_sigma - current_sigma,
            generator=None,
            step_index=step_index,
            timestep=replay_timesteps[step_index],
        )
        rows.copy_(prev_sample[0])

    return step


def _replay_timesteps(sigmas: list[float]) -> torch.Tensor:
    # miles_diffusion's H3 replay reads the trajectory timestep back as sigma.
    return torch.tensor(sigmas[:-1], dtype=torch.float32) * 1000.0


def _cut_at_stop_step(sigmas: list[float], stop_step_index: int | None) -> list[float]:
    if stop_step_index is None:
        return sigmas
    return truncate_rollout_sigmas(sigmas, stop_step_index=stop_step_index)


def _grid_holder(
    sigmas: list[float], replay_timesteps: torch.Tensor
) -> FlowMatchEulerDiscreteScheduler:
    """A scheduler that only carries the grid for the rollout hooks; the H3 loop
    never steps it."""
    holder = FlowMatchEulerDiscreteScheduler()
    holder.sigmas = torch.tensor(sigmas, dtype=torch.float32)
    holder.timesteps = replay_timesteps
    return holder


def prepare_minimax_h3_rollout(
    batch: Req,
    *,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    device: torch.device,
    pipeline_config: Any,
) -> MiniMaxH3RolloutPlan:
    """Prepare both streams before anything reads the step count.

    ``initial_*_rows`` are the target rows only, ``[n, C]``.
    """
    if not batch.rollout:
        return MiniMaxH3RolloutPlan(
            sigmas_video=sigmas_video, sigmas_audio=sigmas_audio
        )
    # The loop and the ODE updates run on the cut grids; the hooks take the
    # uncut ones (sigma_max comes from the full grid) and cut their own.
    cut_video = _cut_at_stop_step(sigmas_video, batch.rollout_stop_step_index)
    cut_audio = _cut_at_stop_step(sigmas_audio, batch.rollout_stop_step_index)
    streams = (
        (
            PRIMARY_STREAM,
            sigmas_video,
            cut_video,
            initial_video_rows,
            _minimax_h3_float_step_tensors(cut_video, device=device),
        ),
        (
            AUDIO_STREAM,
            sigmas_audio,
            cut_audio,
            initial_audio_rows,
            _minimax_h3_step_tensors(cut_audio, device=device),
        ),
    )
    shard_for_sp, gather_for_sp = pipeline_config_sp_fns(pipeline_config)
    for stream, sigmas, cut_sigmas, rows, ode_step_tensors in streams:
        replay_timesteps = _replay_timesteps(sigmas)
        prepare_rollout_if_enabled(
            batch=batch,
            stream=stream,
            scheduler=_grid_holder(sigmas, replay_timesteps),
            timesteps=replay_timesteps,
            latents=rows[None],
            shard_for_sp=shard_for_sp,
            gather_for_sp=gather_for_sp,
            ode_transition=partial(
                _minimax_h3_ode_update,
                sigmas=cut_sigmas,
                step_tensors=ode_step_tensors,
            ),
        )
    # The hook wrote the replay timesteps there; republish H3's own DiT times.
    batch.sigmas = list(cut_video)
    batch.timesteps = torch.tensor(
        [1.0 - sigma for sigma in cut_video[:-1]], dtype=torch.float32
    )
    return MiniMaxH3RolloutPlan(
        sigmas_video=cut_video,
        sigmas_audio=cut_audio,
        video_rollout_step=_stream_rollout_step(
            batch, PRIMARY_STREAM, cut_video, device=device
        ),
        audio_rollout_step=_stream_rollout_step(
            batch, AUDIO_STREAM, cut_audio, device=device
        ),
    )


def _stream_rollout_step(
    batch: Req, stream: str, sigmas: list[float], *, device: torch.device
) -> MiniMaxH3RolloutStep | None:
    transition = rollout_step_kwargs(batch, stream=stream).get("transition")
    if transition is None:
        return None
    return _rollout_step(
        transition=transition,
        sigmas=sigmas,
        replay_timesteps=_replay_timesteps(sigmas),
        device=device,
    )


def finalize_minimax_h3_rollout(
    batch: Req,
    *,
    rollout_plan: MiniMaxH3RolloutPlan,
    video_target_rows: torch.Tensor,
    audio_target_rows: torch.Tensor,
    pipeline_config: Any,
) -> None:
    for stream, sigmas, rows in (
        (PRIMARY_STREAM, rollout_plan.sigmas_video, video_target_rows),
        (AUDIO_STREAM, rollout_plan.sigmas_audio, audio_target_rows),
    ):
        finalize_rollout_if_enabled(
            batch=batch,
            stream=stream,
            latents=rows[None],
            num_inference_steps=len(sigmas) - 1,
            final_timestep=torch.zeros((), dtype=torch.float32),
        )
    publish_rollout_if_enabled(batch=batch, pipeline_config=pipeline_config)


def minimax_h3_denoising_env_kwargs(
    *,
    text_embeddings: torch.Tensor,
    packed: dict[str, Any],
    token_tags: torch.Tensor,
    video_target_start: int,
    audio_target_start: int,
    reference_visual_rows: torch.Tensor | None,
    reference_audio_rows: torch.Tensor | None,
    imgvid_noise_aug: float,
    audio_noise_aug: float,
) -> dict[str, Any]:
    """The cond branch's env, in the names miles_diffusion's H3 replay reads.

    ``reference_*_rows`` are the noised rows the loop pins every step. A layout
    that pins condition rows exports both kinds, an absent kind as an empty
    tensor: the replay writes both whenever it sees one.
    """
    env = {
        "encoder_hidden_states": text_embeddings,
        "h3_packed_layout": packed,
        "h3_token_tags": token_tags,
        "h3_video_target_start": video_target_start,
    }
    if video_target_start == 0 and audio_target_start == 0:
        return env
    env.update(
        h3_reference_visual=(
            reference_visual_rows
            if reference_visual_rows is not None
            else torch.empty(0, MINIMAX_H3_VIDEO_ROW_WIDTH)
        ),
        h3_reference_audio=(
            reference_audio_rows
            if reference_audio_rows is not None
            else torch.empty(0, MINIMAX_H3_AUDIO_ROW_WIDTH)
        ),
        h3_reference_visual_timestep=imgvid_noise_aug,
        h3_reference_audio_timestep=audio_noise_aug,
    )
    return env


__all__ = [
    "MiniMaxH3RolloutPlan",
    "finalize_minimax_h3_rollout",
    "minimax_h3_denoising_env_kwargs",
    "prepare_minimax_h3_rollout",
]
