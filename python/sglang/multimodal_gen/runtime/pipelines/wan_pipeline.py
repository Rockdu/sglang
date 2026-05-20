# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanTimestepPreparationStage(TimestepPreparationStage):
    def forward(self, batch, server_args):
        self.scheduler.prepare_for_batch(batch)
        return super().forward(batch, server_args)


class WanRolloutScheduler(SchedulerRLMixin):
    """Use UniPC for normal Wan inference and Euler SDE for RL rollout."""

    def __init__(self, shift: float | None):
        self.unipc_scheduler = FlowUniPCMultistepScheduler(shift=shift)
        self.euler_scheduler = FlowMatchEulerDiscreteScheduler(
            shift=1.0 if shift is None else shift
        )
        self._active_scheduler = self.unipc_scheduler
        self._logged_rollout_euler_check = False

    def prepare_for_batch(self, batch):
        self._active_scheduler = (
            self.euler_scheduler
            if getattr(batch, "rollout", False)
            else self.unipc_scheduler
        )
        return self._active_scheduler

    @property
    def active_scheduler(self):
        return self._active_scheduler

    @property
    def order(self):
        return self._active_scheduler.order

    @property
    def num_train_timesteps(self):
        return self._active_scheduler.num_train_timesteps

    @property
    def timesteps(self):
        return self._active_scheduler.timesteps

    @property
    def sigmas(self):
        return self._active_scheduler.sigmas

    @property
    def config(self):
        return self._active_scheduler.config

    def __getattr__(self, name):
        return getattr(self._active_scheduler, name)

    def set_shift(self, shift: float) -> None:
        self.unipc_scheduler.set_shift(shift)
        self.euler_scheduler.set_shift(shift)

    def set_begin_index(self, begin_index: int = 0):
        return self._active_scheduler.set_begin_index(begin_index)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device=None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
        **kwargs,
    ):
        if self._active_scheduler is self.unipc_scheduler:
            if timesteps is not None:
                raise ValueError("Wan UniPC scheduler does not support custom timesteps")
            self.unipc_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                **kwargs,
            )
            return

        self.euler_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
            **kwargs,
        )
        self._check_rollout_euler_timesteps()

    def _check_rollout_euler_timesteps(self) -> None:
        sigmas = self.euler_scheduler.sigmas
        timesteps = self.euler_scheduler.timesteps
        if sigmas is None or timesteps is None or sigmas.numel() < 2:
            return
        reconstructed = sigmas[:-1].to(device=timesteps.device) * float(
            self.euler_scheduler.config.num_train_timesteps
        )
        max_abs_diff = (timesteps.float() - reconstructed.float()).abs().max().item()
        if max_abs_diff > 1e-3:
            raise ValueError(
                "Wan rollout Euler timestep/sigma mismatch: "
                f"max_abs_diff={max_abs_diff:.6g}"
            )
        if not self._logged_rollout_euler_check:
            logger.info(
                "Wan rollout using FlowMatchEulerDiscreteScheduler "
                "(timesteps dtype=%s, sigmas dtype=%s, max_abs_diff=%.6g)",
                timesteps.dtype,
                sigmas.dtype,
                max_abs_diff,
            )
            self._logged_rollout_euler_check = True

    def scale_model_input(self, sample, timestep=None):
        return self._active_scheduler.scale_model_input(sample, timestep)

    def step(
        self,
        model_output,
        timestep,
        sample,
        generator=None,
        batch=None,
        return_dict: bool = True,
        **kwargs,
    ):
        if self._active_scheduler is self.unipc_scheduler:
            return self.unipc_scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                generator=generator,
                return_dict=return_dict,
            )
        return self.euler_scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            batch=batch,
            return_dict=return_dict,
            **kwargs,
        )

    def index_for_timestep(self, *args, **kwargs):
        return self._active_scheduler.index_for_timestep(*args, **kwargs)


class WanPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Wan video diffusion pipeline with LoRA support.
    """

    pipeline_name = "WanPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        # We use UniPCMScheduler from Wan2.1 official repo, not the one in diffusers.
        self.modules["scheduler"] = WanRolloutScheduler(
            shift=server_args.pipeline_config.flow_shift
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()
        self.add_stage(WanTimestepPreparationStage(self.get_module("scheduler")))
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = WanPipeline
