"""Mixin for rollout-related denoising hooks.

Moved out of DenoisingStage to keep the core stage lean.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.post_training.sp_utils import (
    gather_stacked_latents_for_sp,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _kwargs_to_cpu(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        elif isinstance(v, (list, tuple)):
            out[k] = [
                t.detach().cpu() if isinstance(t, torch.Tensor) else t for t in v
            ]
        elif isinstance(v, dict):
            out[k] = _kwargs_to_cpu(v)
        else:
            out[k] = v
    return out


class RolloutDenoisingMixin:

    def _maybe_prepare_rollout(self, batch: Req):
        """Prepare denoising loop for rollout."""
        if not isinstance(self.scheduler, SchedulerRLMixin):
            if batch.rollout:
                raise ValueError(
                    f"Scheduler {type(self.scheduler)} does not support rollout"
                )
            return

        self.scheduler.release_rollout_resources(batch)
        if batch.rollout:
            self.scheduler.prepare_rollout(
                batch=batch,
                pipeline_config=self.server_args.pipeline_config,
            )

    def _maybe_collect_rollout_log_probs(self, batch: Req):
        if not isinstance(self.scheduler, SchedulerRLMixin):
            if batch.rollout:
                raise ValueError(
                    f"Scheduler {type(self.scheduler)} does not support rollout"
                )
            return

        if batch.rollout:
            if batch.rollout_trajectory_data is None:
                batch.rollout_trajectory_data = RolloutTrajectoryData()
            batch.rollout_trajectory_data.rollout_log_probs = (
                self.scheduler.collect_rollout_log_probs(batch)
            )
            if batch.rollout_debug_mode:
                batch.rollout_trajectory_data.rollout_debug_tensors = (
                    self.scheduler.collect_rollout_debug_tensors(batch)
                )
            self.scheduler.release_rollout_resources(batch)

    def _postprocess_rollout_outputs(self, batch: Req, server_args: ServerArgs) -> None:
        self._maybe_collect_rollout_log_probs(batch)
        self._maybe_finalize_dit_env_collection(
            batch=batch,
            pipeline_config=server_args.pipeline_config,
        )

    def _maybe_init_denoising_env_collection(
        self,
        batch,
        pipeline_config,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        guidance: torch.Tensor | None,
    ) -> None:
        collect_env = batch.rollout_return_denoising_env
        collect_traj = batch.rollout_return_dit_trajectory
        if not (collect_env or collect_traj):
            batch._rollout_dit_env_state = None
            return

        sanitize = getattr(pipeline_config, "sanitize_dit_env_kwargs", lambda x: x)
        if collect_env:
            env = RolloutDenoisingEnv(
                image_kwargs=_kwargs_to_cpu(sanitize(image_kwargs)),
                pos_cond_kwargs=_kwargs_to_cpu(sanitize(pos_cond_kwargs)),
                neg_cond_kwargs=(
                    _kwargs_to_cpu(sanitize(neg_cond_kwargs))
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
        if state is None or not batch.rollout_return_dit_trajectory:
            return

        state["trajectory_latent_model_inputs"].append(latent_model_input.detach())
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

        if step_inputs and batch.rollout_return_dit_trajectory:
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

        if env is not None and batch.rollout_return_denoising_env:
            sanitize = getattr(pipeline_config, "sanitize_dit_env_kwargs", lambda x: x)
            gather_fn = getattr(pipeline_config, "gather_dit_env_static_for_sp", None)

            pos_src = state.get("pos_cond_kwargs_src")
            if pos_src is not None and env.pos_cond_kwargs is not None:
                gathered_pos = gather_fn(batch, pos_src) if gather_fn else pos_src
                env.pos_cond_kwargs = _kwargs_to_cpu(sanitize(gathered_pos))

            neg_src = state.get("neg_cond_kwargs_src")
            if neg_src is not None and env.neg_cond_kwargs is not None:
                gathered_neg = gather_fn(batch, neg_src) if gather_fn else neg_src
                env.neg_cond_kwargs = _kwargs_to_cpu(sanitize(gathered_neg))

            batch.rollout_trajectory_data.denoising_env = env

        batch._rollout_dit_env_state = None
