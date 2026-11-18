# SPDX-License-Identifier: Apache-2.0
"""RL-specific dataclasses used by post-training and rollout paths."""

from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable

import torch


@dataclass
class RolloutDebugTensors:
    """Container for rollout debug tensors collected during denoising."""

    rollout_variance_noises: torch.Tensor | None = None
    rollout_prev_sample_means: torch.Tensor | None = None
    rollout_noise_std_devs: torch.Tensor | None = None
    rollout_model_outputs: torch.Tensor | None = None


@dataclass
class RolloutDenoisingEnv:
    image_kwargs: dict[str, Any] | None = None
    pos_cond_kwargs: dict[str, Any] | None = None
    neg_cond_kwargs: dict[str, Any] | None = None
    guidance: torch.Tensor | None = None


@dataclass
class RolloutDitTrajectory:
    # [B, T+1, ...]: per-step noisy latents x_{t_0..t_{T-1}} followed by the
    # final denoised latent x_{t_T} (last scheduler.step output).
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None  # [T+1], includes the terminal timestep
    # [T+1] scheduler.sigmas snapshot (post-shift, includes terminal 0).
    sigmas: torch.Tensor | None = None
    # [K] original step index of each kept latent (0..T); None means the full
    # 0..T trajectory. Set whenever rollout_return_step_indices filtered it.
    latent_step_indices: torch.Tensor | None = None


@dataclass
class RolloutStreamData:
    rollout_log_probs: torch.Tensor | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    dit_trajectory: RolloutDitTrajectory | None = None


@dataclass
class RolloutTrajectoryData:
    """The replay env shared by every stream, and each stream's rollout data by name."""

    denoising_env: RolloutDenoisingEnv | None = None
    streams: dict[str, RolloutStreamData] = field(default_factory=dict)


def map_stream_tensors(
    stream: RolloutStreamData | None,
    fn: Callable[[torch.Tensor | None], torch.Tensor | None],
) -> RolloutStreamData | None:
    """Apply ``fn`` to every per-sample tensor of one stream; the schedule is shared."""
    if stream is None:
        return None
    debug = stream.rollout_debug_tensors
    trajectory = stream.dit_trajectory
    return RolloutStreamData(
        rollout_log_probs=fn(stream.rollout_log_probs),
        rollout_debug_tensors=(
            RolloutDebugTensors(
                **{
                    field.name: fn(getattr(debug, field.name))
                    for field in fields(RolloutDebugTensors)
                }
            )
            if debug is not None
            else None
        ),
        dit_trajectory=(
            replace(trajectory, latents=fn(trajectory.latents))
            if trajectory is not None
            else None
        ),
    )


def _cat_per_output(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Concatenate complete per-output tensor fields along the batch dimension."""
    if not tensors or any(tensor is None for tensor in tensors):
        return None
    return torch.cat(tensors, dim=0)


def _concat_stream_data(
    streams: list[RolloutStreamData | None],
) -> RolloutStreamData | None:
    if not streams or any(stream is None for stream in streams):
        return None
    debug = None
    if all(stream.rollout_debug_tensors is not None for stream in streams):
        debug = RolloutDebugTensors(
            **{
                field.name: _cat_per_output(
                    [
                        getattr(stream.rollout_debug_tensors, field.name)
                        for stream in streams
                    ]
                )
                for field in fields(RolloutDebugTensors)
            }
        )
    trajectory = None
    if all(stream.dit_trajectory is not None for stream in streams):
        trajectory = replace(
            streams[0].dit_trajectory,
            latents=_cat_per_output(
                [stream.dit_trajectory.latents for stream in streams]
            ),
        )
    return RolloutStreamData(
        rollout_log_probs=_cat_per_output(
            [stream.rollout_log_probs for stream in streams]
        ),
        rollout_debug_tensors=debug,
        dit_trajectory=trajectory,
    )


def concat_rollout_trajectory_data(
    per_output: list[RolloutTrajectoryData | None],
) -> RolloutTrajectoryData | None:
    """Combine aligned per-output trajectories into one batched trajectory."""
    if not per_output or any(data is None for data in per_output):
        return None
    if len(per_output) == 1:
        return per_output[0]
    names = per_output[0].streams.keys()
    if any(data.streams.keys() != names for data in per_output):
        raise ValueError("Outputs of one request must roll out the same streams")
    return RolloutTrajectoryData(
        denoising_env=per_output[0].denoising_env,
        streams={
            name: _concat_stream_data([data.streams[name] for data in per_output])
            for name in names
        },
    )


def _slice_output_dim(
    tensor: torch.Tensor | None, output_index: int
) -> torch.Tensor | None:
    """Keep one output row while preserving the batch dimension."""
    if tensor is None or tensor.dim() < 1 or output_index >= tensor.shape[0]:
        return tensor
    return tensor[output_index : output_index + 1]


def select_output_rollout_trajectory(
    rollout_trajectory_data: RolloutTrajectoryData | None,
    output_index: int | None,
) -> RolloutTrajectoryData | None:
    """Select one output row from a batched rollout trajectory."""
    if rollout_trajectory_data is None or output_index is None:
        return rollout_trajectory_data

    def select(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return _slice_output_dim(tensor, output_index)

    return RolloutTrajectoryData(
        denoising_env=rollout_trajectory_data.denoising_env,
        streams={
            name: map_stream_tensors(stream, select)
            for name, stream in rollout_trajectory_data.streams.items()
        },
    )
