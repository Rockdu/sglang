# SPDX-License-Identifier: Apache-2.0
"""Rollout / RL hooks for FLUX pipeline configs."""

from __future__ import annotations


class FluxRolloutPipelineMixin:
    # TODO: Flux PE backpasses
    """``gather_dit_env_static_for_sp``: FLUX interleaves txt+img in cos/sin; TBD split/gather/reconcat."""

    def gather_dit_env_static_for_sp(self, batch, cond_kwargs: dict | None):
        return cond_kwargs
