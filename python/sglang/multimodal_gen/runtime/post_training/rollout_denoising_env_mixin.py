"""Mixin for collecting DiT denoising environment during rollout.

``RolloutDenoisingEnv`` (dit env: image / cond kwargs / guidance) is filled when
``batch.rollout_return_dit_env`` is set. Per-step inputs go to
``RolloutDitTrajectory`` when ``batch.rollout_return_dit_trajectory`` is set.
Either or both may be enabled independently.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_latents_for_sp,
)
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)


class RolloutDenoisingEnvMixin:
    """Collect and finalize rollout DiT environment with minimal stage coupling."""

    @staticmethod
    def _kwargs_to_cpu(d: dict[str, Any]) -> dict[str, Any]:
        """Deep-copy kwargs dict, moving tensors to CPU recursively."""
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu()
            elif isinstance(v, (list, tuple)):
                out[k] = [
                    t.detach().cpu() if isinstance(t, torch.Tensor) else t for t in v
                ]
            elif isinstance(v, dict):
                out[k] = RolloutDenoisingEnvMixin._kwargs_to_cpu(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _should_collect_dit_env(batch) -> bool:
        return bool(getattr(batch, "rollout_return_dit_env", False))

    @staticmethod
    def _should_collect_dit_trajectory(batch) -> bool:
        return bool(getattr(batch, "rollout_return_dit_trajectory", False))

    @staticmethod
    def _should_init_dit_collection(batch) -> bool:
        return RolloutDenoisingEnvMixin._should_collect_dit_env(
            batch
        ) or RolloutDenoisingEnvMixin._should_collect_dit_trajectory(batch)

    @staticmethod
    def _call_gather_dit_env_static_for_sp_if_defined(
        pipeline_config: Any,
        batch: Any,
        cond_kwargs: dict | None,
    ) -> dict | None:
        """If ``pipeline_config`` defines ``gather_dit_env_static_for_sp``, call it; else no-op."""
        fn = getattr(pipeline_config, "gather_dit_env_static_for_sp", None)
        if fn is not None and callable(fn):
            return fn(batch, cond_kwargs)
        return cond_kwargs

    def _maybe_init_dit_env_collection(
        self,
        batch,
        pipeline_config,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        guidance: torch.Tensor | None,
    ) -> None:
        if not self._should_init_dit_collection(batch):
            batch._rollout_dit_env_state = None
            return

        sanitize = getattr(pipeline_config, "sanitize_dit_env_kwargs", lambda x: x)
        if self._should_collect_dit_env(batch):
            env = RolloutDenoisingEnv(
                image_kwargs=self._kwargs_to_cpu(sanitize(image_kwargs)),
                pos_cond_kwargs=self._kwargs_to_cpu(sanitize(pos_cond_kwargs)),
                neg_cond_kwargs=(
                    self._kwargs_to_cpu(sanitize(neg_cond_kwargs))
                    if neg_cond_kwargs
                    else None
                ),
                guidance=guidance.detach().cpu() if guidance is not None else None,
            )
            pos_src = pos_cond_kwargs
            neg_src = neg_cond_kwargs
        else:
            env = None
            pos_src = None
            neg_src = None

        batch._rollout_dit_env_state = {
            "env": env,
            "trajectory_latent_model_inputs": [],
            "trajectory_timesteps": [],
            # Device-side cond kwargs for optional SP gather at finalize (do not mutate).
            "pos_cond_kwargs_src": pos_src,
            "neg_cond_kwargs_src": neg_src,
        }

    def _maybe_append_dit_env_step(
        self,
        batch,
        latent_model_input: torch.Tensor,
        timestep_value: torch.Tensor,
    ) -> None:
        state = getattr(batch, "_rollout_dit_env_state", None)
        if state is None or not self._should_collect_dit_trajectory(batch):
            return

        state["trajectory_latent_model_inputs"].append(latent_model_input.detach())
        # Use scalar timestep values to avoid storing expanded/sharded timestep tensors.
        state["trajectory_timesteps"].append(timestep_value.detach().cpu())

    def _maybe_finalize_dit_env_collection(self, batch, pipeline_config) -> None:
        state = getattr(batch, "_rollout_dit_env_state", None)
        if state is None:
            return

        env: RolloutDenoisingEnv | None = state["env"]
        step_inputs: list[torch.Tensor] = state["trajectory_latent_model_inputs"]
        step_timesteps: list[torch.Tensor] = state["trajectory_timesteps"]

        if batch.rollout_trajectory_data is None:
            batch.rollout_trajectory_data = RolloutTrajectoryData()

        if step_inputs and self._should_collect_dit_trajectory(batch):
            # [B, T, ...] — matches rollout_log_probs time dimension
            step_inputs_tensor = torch.stack(step_inputs, dim=1)
            step_inputs_tensor = gather_stacked_latents_for_sp(
                pipeline_config=pipeline_config,
                batch=batch,
                stacked_latents=step_inputs_tensor,
            )
            batch.rollout_trajectory_data.dit_trajectory = RolloutDitTrajectory(
                latent_model_inputs=step_inputs_tensor.cpu(),
                timesteps=torch.stack(step_timesteps, dim=0).cpu(),
            )

        if env is not None and self._should_collect_dit_env(batch):
            sanitize = getattr(pipeline_config, "sanitize_dit_env_kwargs", lambda x: x)

            pos_src = state.get("pos_cond_kwargs_src")
            if pos_src is not None and env.pos_cond_kwargs is not None:
                gathered_pos = self._call_gather_dit_env_static_for_sp_if_defined(
                    pipeline_config, batch, pos_src
                )
                env.pos_cond_kwargs = self._kwargs_to_cpu(sanitize(gathered_pos))

            neg_src = state.get("neg_cond_kwargs_src")
            if neg_src is not None and env.neg_cond_kwargs is not None:
                gathered_neg = self._call_gather_dit_env_static_for_sp_if_defined(
                    pipeline_config, batch, neg_src
                )
                env.neg_cond_kwargs = self._kwargs_to_cpu(sanitize(gathered_neg))

            batch.rollout_trajectory_data.denoising_env = env

        batch._rollout_dit_env_state = None
