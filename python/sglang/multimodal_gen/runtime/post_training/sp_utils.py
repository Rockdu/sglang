# SPDX-License-Identifier: Apache-2.0
"""Shared Sequence Parallel helpers for post-training rollout code."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_reduce,
)


def should_do_sp_collective(batch) -> bool:
    """Return whether SP collectives should run for this batch."""
    return get_sp_world_size() > 1 and getattr(batch, "did_sp_shard_latents", False)


def gather_stacked_latents_for_sp(
    pipeline_config,
    batch,
    stacked_latents: torch.Tensor,
) -> torch.Tensor:
    """Gather a stacked ``[B, T, ...]`` latent tensor via pipeline-config SP rules."""
    if not should_do_sp_collective(batch):
        return stacked_latents

    if stacked_latents.dim() < 2:
        return stacked_latents

    bsz, t_steps = stacked_latents.shape[0], stacked_latents.shape[1]
    flat_inputs = stacked_latents.flatten(0, 1).contiguous()
    gathered_flat_inputs = pipeline_config.gather_latents_for_sp(flat_inputs)
    return gathered_flat_inputs.unflatten(0, (bsz, t_steps))


def all_reduce_if_sp_sharded(batch, tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor in-place on SP group if latents are sharded."""
    if not should_do_sp_collective(batch):
        return tensor
    tensor = tensor.to(get_local_torch_device())
    sequence_model_parallel_all_reduce(tensor)
    return tensor
