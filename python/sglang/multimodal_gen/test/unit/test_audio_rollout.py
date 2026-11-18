# SPDX-License-Identifier: Apache-2.0
"""Regressions specific to audio rollout wiring and request ownership."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from sglang.multimodal_gen.configs.post_training.rl_rollout import AUDIO_STREAM
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DenoisingStage,
)
from sglang.multimodal_gen.runtime.post_training.flow_rollout_controller import (
    truncate_rollout_sigmas,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import (
    PRIMARY_STREAM,
    derive_stream_generators,
    finalize_rollout_if_enabled,
    loop_latent_sp_fns,
    prepare_rollout_if_enabled,
    rollout_scope,
    set_rollout_early_stop_if_enabled,
)

_CONFIG = SimpleNamespace(shard_latents_for_sp=lambda batch, latents: (latents, False))


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
        lambda: 1,
    )
    req = Req(
        sampling_params=SamplingParams(
            prompt="test",
            seed=10,
            rollout=True,
            rollout_extra_streams=(AUDIO_STREAM,),
            rollout_return_dit_trajectory=True,
            rollout_return_step_indices=[0, 2],
        )
    )
    req.scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    req.scheduler.set_timesteps(sigmas=[0.8, 0.4])
    req.timesteps = req.scheduler.timesteps
    req.latents = torch.zeros(2, 4, 2)
    req.audio_latents = torch.ones(2, 5, 3)
    req.generator = [torch.Generator().manual_seed(10 + i) for i in range(2)]
    return req


def _prepare_primary_stream(batch):
    """What a denoising stage does for the stream its loop iterates."""
    shard_for_sp, gather_for_sp = loop_latent_sp_fns(_CONFIG)
    prepare_rollout_if_enabled(
        batch=batch,
        stream=PRIMARY_STREAM,
        scheduler=batch.scheduler,
        timesteps=batch.timesteps,
        latents=batch.latents,
        shard_for_sp=shard_for_sp,
        gather_for_sp=gather_for_sp,
    )


def test_audio_rng_is_derived_from_the_seed_alone(batch):
    """An extra stream's seed depends only on the request seed and the stream name."""
    states = [g.get_state() for g in batch.generator]
    audio = derive_stream_generators(batch.generator, stream=AUDIO_STREAM)
    assert audio[0].initial_seed() == 635590486471046510
    action = derive_stream_generators(batch.generator, stream="action")
    seeds = [
        {g.initial_seed() for g in gens} for gens in (batch.generator, audio, action)
    ]
    assert sum(map(len, seeds)) == len(set().union(*seeds))
    for visual, state in zip(batch.generator, states):
        torch.testing.assert_close(visual.get_state(), state, rtol=0, atol=0)


def _identity_for_sp(batch, tensor):
    return tensor


def _prepare_stream_on_its_own_grid(batch, name):
    """A FlowMatch Euler stream with the primary's step count but another shift."""
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    scheduler.set_timesteps(sigmas=[0.8, 0.4])
    uncut_sigmas = scheduler.sigmas.clone()
    prepare_rollout_if_enabled(
        batch=batch,
        stream=name,
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        latents=batch.audio_latents,
        shard_for_sp=_identity_for_sp,
        gather_for_sp=_identity_for_sp,
    )
    return scheduler, uncut_sigmas


def test_unlisted_stream_cuts_its_own_grid_at_the_stop(batch):
    """A stream that is not rolled out still ends on its own terminal sigma."""
    batch.rollout_extra_streams = ()
    batch.rollout_stop_step_index = 0
    _prepare_primary_stream(batch)
    set_rollout_early_stop_if_enabled(batch=batch)
    scheduler, uncut_sigmas = _prepare_stream_on_its_own_grid(batch, "action")
    assert "action" not in batch.rollout_controllers
    torch.testing.assert_close(
        scheduler.sigmas,
        truncate_rollout_sigmas(uncut_sigmas, stop_step_index=0),
        rtol=0,
        atol=0,
    )
    assert len(scheduler.timesteps) == 1


def test_extra_stream_with_another_step_count_is_rejected(batch):
    batch.rollout_extra_streams = ("action",)
    _prepare_primary_stream(batch)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(sigmas=[0.9, 0.6, 0.3])
    with pytest.raises(ValueError, match="streams step together"):
        prepare_rollout_if_enabled(
            batch=batch,
            stream="action",
            scheduler=scheduler,
            timesteps=scheduler.timesteps,
            latents=batch.audio_latents,
            shard_for_sp=_identity_for_sp,
            gather_for_sp=_identity_for_sp,
        )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (
            {"rollout": False, "rollout_extra_streams": ("audio",)},
            "requires rollout=True",
        ),
        ({"rollout_extra_streams": ("audio", "audio")}, "duplicates"),
        ({"rollout_extra_streams": (PRIMARY_STREAM,)}, "Invalid rollout extra stream"),
        ({"rollout_extra_streams": ("a.b",)}, "Invalid rollout extra stream"),
        ({"rollout_extra_streams": "audio"}, "must be a list"),
    ],
)
def test_extra_stream_names_are_checked_with_the_sampling_params(kwargs, error):
    """Stream names become store path segments, so they are checked up front."""
    with pytest.raises(ValueError, match=error):
        SamplingParams(prompt="test", **{"rollout": True, **kwargs})


def test_batched_noise_rows_match_single_sample_draws(batch):
    """With B>1 each row's noise is its own seed's B=1 draw."""
    _prepare_primary_stream(batch)
    controller = batch.rollout_controllers[PRIMARY_STREAM]
    singles = [torch.Generator().manual_seed(10 + i) for i in range(2)]
    for _ in range(2):
        expected = torch.cat([torch.randn(1, 4, 2, generator=g) for g in singles])
        actual = controller._rollout_variance_noise(batch, batch.latents)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_generators_fill_cuda_noise_for_each_sample(batch):
    _prepare_primary_stream(batch)
    controller = batch.rollout_controllers[PRIMARY_STREAM]
    expected = torch.cat(
        [
            torch.randn(1, 4, 2, generator=torch.Generator().manual_seed(10 + i))
            for i in range(2)
        ]
    )
    actual = controller._rollout_variance_noise(batch, batch.latents.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_generators_fill_cuda_noise_in_place(batch, monkeypatch):
    """A "cuda" generator has no index, so it never compares equal to cuda:0."""
    batch.generator = [torch.Generator("cuda").manual_seed(10 + i) for i in range(2)]
    _prepare_primary_stream(batch)
    controller = batch.rollout_controllers[PRIMARY_STREAM]
    expected = torch.cat(
        [
            torch.randn(
                1,
                4,
                2,
                device="cuda",
                generator=torch.Generator("cuda").manual_seed(10 + i),
            )
            for i in range(2)
        ]
    )
    monkeypatch.setattr(
        torch.Tensor,
        "copy_",
        lambda *args, **kwargs: pytest.fail("same-device noise must not be copied"),
    )
    actual = controller._rollout_variance_noise(batch, batch.latents.cuda())
    monkeypatch.undo()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "stage_cls,loop_name",
    [(DenoisingStage, "_denoise"), (Cosmos3DenoisingStage, "_denoise_once")],
)
def test_stage_cleanup_allows_retry(batch, stage_cls, loop_name):
    """A failed stage run leaves no rollout state, so a retry can prepare again."""
    stage = object.__new__(stage_cls)
    stage._offload_for_torch_compile_warmup = lambda batch: nullcontext()
    stage.use_declared_component = lambda **kwargs: nullcontext()
    stage.transformer = Mock()
    fail = True

    def loop(batch, args):
        _prepare_primary_stream(batch)
        batch._rollout_denoising_env_state = {"env": None}
        batch.rollout_trajectory_data = object()
        if fail:
            raise RuntimeError("model failed")
        return batch

    setattr(stage, loop_name, loop)
    with pytest.raises(RuntimeError, match="model failed"):
        stage.forward(batch, None)
    assert batch.rollout_controllers == {}
    assert batch._rollout_denoising_env_state is None
    assert batch.rollout_trajectory_data is None
    fail = False
    assert stage.forward(batch, None) is batch
    assert batch.rollout_controllers == {}


def test_requested_stream_the_stage_never_prepared_fails_the_request(batch):
    """A listed stream must not silently vanish from the response."""
    batch.rollout_extra_streams = ("action",)
    with pytest.raises(RuntimeError, match="were not prepared"):
        with rollout_scope(batch):
            _prepare_primary_stream(batch)
            finalize_rollout_if_enabled(
                batch=batch,
                latents={PRIMARY_STREAM: batch.latents},
                num_inference_steps=len(batch.timesteps),
                final_timestep=torch.tensor(0.0),
                pipeline_config=_CONFIG,
            )


def test_extra_stream_needs_a_flow_match_euler_scheduler(batch):
    _prepare_primary_stream(batch)
    with pytest.raises(ValueError, match="FlowMatch Euler"):
        prepare_rollout_if_enabled(
            batch=batch,
            stream="action",
            scheduler=SimpleNamespace(sigmas=batch.scheduler.sigmas),
            timesteps=batch.scheduler.timesteps,
            latents=batch.audio_latents,
            shard_for_sp=_identity_for_sp,
            gather_for_sp=_identity_for_sp,
        )
