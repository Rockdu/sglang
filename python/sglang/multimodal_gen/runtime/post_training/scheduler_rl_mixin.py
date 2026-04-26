# SPDX-License-Identifier: Apache-2.0
"""Flow-matching rollout step utilities for log-prob computation."""

import math
from typing import Any, Union

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutSessionData,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_debug_mixin import (
    SchedulerRLDebugMixin,
)

# Match flow_grpo's `torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))`
_LOG_SQRT_2PI_TENSOR = torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))


class SchedulerRLMixin(SchedulerRLDebugMixin):
    @staticmethod
    def _get_rollout_session_data(batch) -> RolloutSessionData:
        """Return the RolloutSessionData attached to *batch*, or raise if not prepared."""
        rollout_session_data = getattr(batch, "_rollout_session_data", None)
        if rollout_session_data is None:
            raise RuntimeError("prepare_rollout() not called before rollout")
        return rollout_session_data

    def release_rollout_resources(self, batch) -> None:
        """Release rollout-owned resources. Call when denoising ends or before a new rollout."""
        batch._rollout_session_data = None

    def prepare_rollout(self, batch: Req, pipeline_config: Any = None) -> None:
        """Enable rollout and set SDE/CPS params. Call once before the denoising loop."""
        if get_sp_world_size() > 1 and pipeline_config is None:
            raise RuntimeError(
                "SP rollout requires pipeline_config to be passed to prepare_rollout()."
            )
        batch._rollout_session_data = RolloutSessionData(
            pipeline_config=pipeline_config,
            sigma_max=self.sigmas[min(1, len(self.sigmas) - 1)].item(),
            latents_shape=(
                tuple(batch.latents.shape) if batch.latents is not None else None
            ),
        )

    def already_prepared_rollout(self, batch) -> bool:
        return getattr(batch, "_rollout_session_data", None) is not None

    def _get_or_create_rollout_noise_buffer(
        self,
        rollout_session_data: RolloutSessionData,
        full_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or create the reusable noise buffer (local or full shape) for rollout."""
        buffer = rollout_session_data.noise_buffer
        if (
            buffer is None
            or buffer.shape != full_shape
            or buffer.dtype != dtype
            or buffer.device != device
        ):
            buffer = torch.empty(full_shape, device=device, dtype=dtype)
            rollout_session_data.noise_buffer = buffer
        return buffer

    def _rollout_variance_noise(
        self,
        batch,
        model_output: torch.FloatTensor,
        generator: Union[torch.Generator, list[torch.Generator]],
    ) -> torch.FloatTensor:
        """Generate variance noise for rollout. If generator is a list, use generator[i] for the i-th batch item."""
        assert generator is not None, "Generator must be provided"

        rollout_session_data = self._get_rollout_session_data(batch)
        device = model_output.device
        dtype = model_output.dtype
        local_shape = tuple(model_output.shape)

        B = local_shape[0]
        if isinstance(generator, torch.Generator):
            assert B == 1, "Generator must be a list if batch size is not 1"
            generator = [generator]
        else:
            assert (
                len(generator) == B
            ), "Generator list must have the same length as batch size"

        buffer = self._get_or_create_rollout_noise_buffer(
            rollout_session_data, rollout_session_data.latents_shape, device, dtype
        )
        for i in range(B):
            torch.randn(
                rollout_session_data.latents_shape,
                out=buffer[i : i + 1],
                generator=generator[i],
            )

        sharded_noise, _ = rollout_session_data.pipeline_config.shard_latents_for_sp(
            batch=batch, latents=buffer
        )
        if tuple(sharded_noise.shape) != local_shape:
            raise ValueError(
                "Rollout SP noise shape mismatch after shard. "
                f"Expected local_shape={local_shape}, got {tuple(sharded_noise.shape)}."
            )
        return sharded_noise

    def flow_sde_sampling(
        self,
        batch,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        current_sigma: torch.FloatTensor,
        next_sigma: torch.FloatTensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Flow rollout step for log-prob / sampling (see FlowGRPO-style references).

        ``rollout_sde_type`` (from batch SamplingParams):

        1. ``"sde"``: Standard stochastic differential equation transition (Gaussian).
        2. ``"cps"``: Coupled Particle Sampling.
        3. ``"ode"``: Deterministic ODE step (no diffusion noise).
        """
        rollout_session_data = self._get_rollout_session_data(batch)
        sde_type = batch.rollout_sde_type
        noise_level = float(batch.rollout_noise_level)
        log_prob_no_const = batch.rollout_log_prob_no_const
        debug_mode = bool(getattr(batch, "rollout_debug_mode", False))

        if not log_prob_no_const and sde_type != "ode":
            assert (
                noise_level > 0
            ), "True log-probability computation requires a non-zero noise level."

        dt = next_sigma - current_sigma

        # step_index comes from the denoising-loop counter stashed by
        # DenoisingStage — scheduler._step_index would differ when
        # _begin_index != 0 (e.g. partial denoising).
        sde_step_indices = getattr(batch, "rollout_sde_step_indices", None)
        loop_step_index = getattr(batch, "_rollout_loop_step_index", None)
        if (
            sde_type != "ode"
            and sde_step_indices is not None
            and loop_step_index is not None
            and loop_step_index not in sde_step_indices
        ):
            effective_sde_type = "ode"
        else:
            effective_sde_type = sde_type

        # sde/cps: cast to fp32 to match flowGRPO semantics and avoid the
        # 0-dim-fp32 wrapped-scalar promotion demoting log-prob to bf16.
        # ode: keep dtypes unchanged so rollout(ode) stays bit-exact with
        # rollout=False (scheduling_flow_match_euler_discrete.step()).
        # log_prob is computed on the full pre-shard noise buffer so SP ranks
        # produce identical sums — see collect_rollout_log_probs().
        if effective_sde_type == "sde":
            model_output = model_output.float()
            sample = sample.float()
            variance_noise = self._rollout_variance_noise(
                batch, model_output, generator
            )
            full_variance_noise = rollout_session_data.noise_buffer
            # Match flow_grpo `sigma == 1` (sd3_sde_with_logprob.py:50)
            std_dev_t = (
                torch.sqrt(
                    current_sigma
                    / (
                        1
                        - torch.where(
                            current_sigma == 1,
                            rollout_session_data.sigma_max,
                            current_sigma,
                        )
                    )
                )
                * noise_level
            )
            noise_std_dev = std_dev_t * torch.sqrt(-1 * dt)
            prev_sample_mean = (
                sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt)
                + model_output
                * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma))
                * dt
            )

            weighted_variance_noise = variance_noise * noise_std_dev
            prev_sample = prev_sample_mean + weighted_variance_noise
            # Match flow_grpo's `(prev_sample - prev_sample_mean)**2` bit-
            # exactly (sd3_sde_with_logprob.py:65).
            log_prob_no_const_val = -((prev_sample - prev_sample_mean) ** 2)

        elif effective_sde_type == "cps":
            model_output = model_output.float()
            sample = sample.float()
            variance_noise = self._rollout_variance_noise(
                batch, model_output, generator
            )
            full_variance_noise = rollout_session_data.noise_buffer
            std_dev_t = next_sigma * math.sin(noise_level * math.pi / 2)
            noise_std_dev = std_dev_t
            pred_original_sample = sample - current_sigma * model_output
            noise_estimate = sample + model_output * (1 - current_sigma)
            prev_sample_mean = pred_original_sample * (
                1 - next_sigma
            ) + noise_estimate * torch.sqrt(next_sigma**2 - std_dev_t**2)

            weighted_variance_noise = variance_noise * noise_std_dev
            prev_sample = prev_sample_mean + weighted_variance_noise
            # Same rationale as the sde branch above.
            log_prob_no_const_val = -((prev_sample - prev_sample_mean) ** 2)

        elif effective_sde_type == "ode":
            prev_sample = sample + dt * model_output
            prev_sample_mean = prev_sample
            variance_noise = torch.zeros_like(model_output)
            noise_std_dev = torch.zeros(
                (), device=model_output.device, dtype=model_output.dtype
            )
            log_prob_no_const_val = torch.zeros(
                rollout_session_data.latents_shape,
                device=model_output.device,
                dtype=torch.float32,
            )
            # Only enforce the "no full log-prob with ODE" constraint when the
            # user explicitly chose ODE globally.
            if sde_type == "ode":
                assert (
                    log_prob_no_const
                ), "p_ode is always 0, true log_prob is meaningless, set rollout_log_prob_no_const to True to enable log_prob computation"

        else:
            raise ValueError(f"Unsupported sde_type: {sde_type}")

        reduce_dims = list(range(1, len(log_prob_no_const_val.shape)))
        # Match flow_grpo per-element formula then mean along non-batch dims.
        log_const_tensor = _LOG_SQRT_2PI_TENSOR.to(
            log_prob_no_const_val.device, log_prob_no_const_val.dtype
        )

        if log_prob_no_const or effective_sde_type == "ode":
            log_prob_local = log_prob_no_const_val.mean(dim=reduce_dims)
        else:
            log_prob_local = (
                log_prob_no_const_val / (2 * (noise_std_dev**2))
                - torch.log(noise_std_dev)
                - log_const_tensor
            ).mean(dim=reduce_dims)

        if debug_mode:
            self.append_local_rollout_debug_tensors(
                batch,
                variance_noise=variance_noise,
                prev_sample_mean=prev_sample_mean,
                noise_std_dev=noise_std_dev,
                model_output=model_output,
            )

        self.append_local_rollout_log_probs(batch, log_prob_local)

        return prev_sample

    def append_local_rollout_log_probs(
        self, batch, log_prob: torch.Tensor
    ) -> None:
        rollout_session_data = self._get_rollout_session_data(batch)
        rollout_session_data.local_log_probs.append(log_prob)

    def consume_local_rollout_log_probs(
        self, batch
    ) -> torch.Tensor:
        rollout_session_data = self._get_rollout_session_data(batch)
        log_probs = torch.stack(rollout_session_data.local_log_probs, dim=1)
        rollout_session_data.local_log_probs = []
        return log_probs

    def collect_rollout_log_probs(self, batch: Req) -> torch.Tensor | None:
        trajectory_log_probs = self.consume_local_rollout_log_probs(batch)
        return trajectory_log_probs.cpu()
