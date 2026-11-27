# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts of the H3 denoise loop: request-static metadata, the
serving target-row update, and the rollout through the shared controllers."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.configs.post_training.rl_rollout import (
    AUDIO_STREAM,
    PRIMARY_STREAM,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _build_local_embedding_layout,
    _minimax_h3_update_target_rows_,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.minimax_h3_rollout import (
    finalize_minimax_h3_rollout,
    prepare_minimax_h3_rollout,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
    _build_cube_attn_metadata,
    _precompute_refined_prompt_embeds,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.post_training.rollout_hooks import rollout_scope

_SP_IDENTITY_PIPELINE_CONFIG = SimpleNamespace(
    shard_latents_for_sp=lambda batch, latents: (latents, False)
)


def _minimax_h3_rf_v_to_x0(
    xt: torch.Tensor,
    v: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    cond_t = timestep.to(device=xt.device, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    sigma_t = 1 - cond_t
    return xt + sigma_t * v


def _minimax_h3_euler_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_ratio: torch.Tensor,
) -> torch.Tensor:
    if sigma_curr == 0.0:
        return state
    compute_dtype = torch.float32
    if state.dtype not in (torch.float16, torch.bfloat16):
        compute_dtype = state.dtype
    ratio = sigma_ratio.to(device=state.device, dtype=compute_dtype)
    out = ratio * state.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(
        dtype=compute_dtype
    )
    return out.to(dtype=state.dtype)


def _branch(
    mode: str, token_tags: torch.Tensor | None = None
) -> MiniMaxH3DenoiseBranch:
    common = dict(text_len=3, latent_t=2, latent_h=4, latent_w=4, audio_t=3)
    if mode == "t2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=False,
        )
    elif mode == "fl2va":
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
        )
    else:
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        )
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"] if token_tags is None else token_tags,
        device=torch.device("cpu"),
    )


def test_time_shift_sigmas_build_exact_denoise_step_count():
    for num_steps in (1, 2, 10, 50):
        for shift_scale in (3.0, 12.0):
            sigmas = minimax_h3_time_shift_sigmas(
                num_steps=num_steps,
                shift_scale=shift_scale,
            )

            assert len(sigmas) == num_steps + 1
            assert sigmas[0] == 1.0
            assert sigmas[-1] == 0.0
            assert all(
                current > following for current, following in zip(sigmas, sigmas[1:])
            )
            # Independent closed form for N Euler intervals, including sigma=0.
            expected = [
                shift_scale
                * (num_steps - i)
                / (num_steps + (shift_scale - 1) * (num_steps - i))
                for i in range(num_steps + 1)
            ]
            torch.testing.assert_close(
                torch.tensor(sigmas), torch.tensor(expected), rtol=0, atol=2e-7
            )


def test_precomputed_timestep_plan_matches_full_unique_reference():
    """Preplanning must preserve fp32 collisions and every packed row class."""

    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        assert branch.static_kwargs["skip_mask_out_condition"]
        assert "token_tags" not in branch.static_kwargs
        assert not bool((branch.static_kwargs["block_token_tags"] < 0).any())
        torch.testing.assert_close(
            branch.static_kwargs["img_pos_for_infer_output_info"]["position_ids"],
            branch.img_pos_dev[branch.update_mask_dev],
            rtol=0,
            atol=0,
        )
        video_steps = [0.75, 0.1]
        audio_steps = [0.625, 0.2]
        plan = branch.prepare_timestep_plan(
            video_timesteps=video_steps,
            audio_timesteps=audio_steps,
            imgvid_cond_noise_aug=0.6,
            audio_ref_cond_noise_aug=0.4,
        )

        assert branch.static_kwargs["packed_seq_params"]["cu_seqlens_q_host"] == tuple(
            int(value)
            for value in branch.static_kwargs["packed_seq_params"][
                "cu_seqlens_q"
            ].tolist()
        )
        assert branch.static_kwargs["refiner_packed_seq_params"][
            "cu_seqlens_q_host"
        ] == (0, 3, 3)

        for step, (video_t, audio_t) in enumerate(
            zip(video_steps, audio_steps, strict=True)
        ):
            reference = torch.full((branch.seq_len,), video_t, dtype=torch.float32)
            reference[branch.img_cond_seq_idx] = max(video_t, 0.6)
            reference[branch.audio_target_seq_idx] = audio_t
            reference[branch.audio_ref_seq_idx] = max(audio_t, 0.4)
            expected = torch.unique(reference, sorted=True, return_inverse=True)
            torch.testing.assert_close(plan[step][0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(plan[step][1], expected[1], rtol=0, atol=0)
            torch.testing.assert_close(
                plan[step][2],
                branch.static_kwargs["block_token_tags"]
                + expected[1] * MINIMAX_H3_ADALN_MODALITY_NUM,
                rtol=0,
                atol=0,
            )

        repeated_plan = branch.prepare_timestep_plan(
            video_timesteps=[0.0, 0.1, 0.2],
            audio_timesteps=[0.0, 0.2, 0.4],
            imgvid_cond_noise_aug=0.999,
            audio_ref_cond_noise_aug=1.0,
        )
        assert repeated_plan[1][1] is repeated_plan[2][1]
        assert repeated_plan[1][2] is repeated_plan[2][2]


def test_inplace_target_update_matches_scheduler_math():
    generator = torch.Generator().manual_seed(7)
    for sigma_curr, sigma_next in ((1.0, 0.7), (0.2, 0.0), (0.0, 0.0)):
        state = torch.randn(11, 32, generator=generator)
        velocity = torch.randn(11, 32, generator=generator)
        timestep = torch.tensor(1.0 - sigma_curr)
        ratio = torch.tensor(0.0 if sigma_curr == 0.0 else sigma_next / sigma_curr)
        denoised = _minimax_h3_rf_v_to_x0(state, velocity, timestep)
        expected = _minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
        )

        actual = state.clone()
        velocity_scratch = velocity.clone()
        _minimax_h3_update_target_rows_(
            actual,
            velocity_scratch,
            sigma_t=1.0 - timestep,
            sigma_curr=sigma_curr,
            sigma_ratio=ratio,
            one_minus_sigma_ratio=1.0 - ratio,
            denoised_scratch=torch.empty_like(actual),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_text_layout_is_a_contiguous_prefix_per_ulysses_rank():
    for mode in ("t2va", "fl2va", "ref2va"):
        branch = _branch(mode)
        text_len = int(branch.static_kwargs["prompt_embeds"].shape[0])
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                layout = _build_local_embedding_layout(
                    seq_len=branch.seq_len,
                    text_pos=torch.arange(text_len),
                    img_pos=branch.img_pos,
                    audio_pos=branch.audio_pos,
                    world_size=world_size,
                    rank=rank,
                    device=torch.device("cpu"),
                )
                start = int(layout["text_source_start"])
                stop = int(layout["text_source_stop"])
                row_start = rank * (branch.seq_len // world_size)
                expected = torch.nonzero(
                    (torch.arange(text_len) >= row_start)
                    & (
                        torch.arange(text_len)
                        < row_start + branch.seq_len // world_size
                    )
                ).view(-1)
                assert expected.tolist() == list(range(start, stop))


def test_rank_local_token_tags_match_reference_slice():
    for mode in ("t2va", "fl2va", "ref2va"):
        seq_len = _branch(mode).seq_len
        token_tags = torch.arange(seq_len, dtype=torch.long) - seq_len // 2
        for world_size in (1, 2, 4, 8):
            for rank in range(world_size):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.stages."
                    "model_specific_stages.minimax_h3.denoise_loop.get_ulysses_ctx",
                    return_value=(world_size, rank),
                ):
                    branch = _branch(mode, token_tags=token_tags)
                local_rows = branch.seq_len // world_size
                expected = token_tags[
                    rank * local_rows : (rank + 1) * local_rows
                ].clamp(min=0)
                torch.testing.assert_close(
                    branch.static_kwargs["block_token_tags"], expected, rtol=0, atol=0
                )


def test_cube_metadata_builder_uses_packed_layout_and_validates_step_count():
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=3,
        include_keyframe_cond=False,
    )
    server_args = SimpleNamespace(
        attention_backend="cube_sparse_attn",
        component_attention_backends={},
        attention_backend_config={
            "local_cube_size": [4, 4, 4],
            "topk_ratio_list": [1.0, 0.5],
        },
    )

    metadata = _build_cube_attn_metadata(
        server_args,
        packed=packed,
        num_steps=2,
        device=torch.device("cpu"),
    )
    assert metadata.topk_ratio_list == [1.0, 0.5]
    assert metadata.precomputed.layout.cube_token_size == 64

    with pytest.raises(ValueError, match="denoise steps"):
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=3,
            device=torch.device("cpu"),
        )


def test_cube_metadata_follows_transformer_backend_override():
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=3,
        include_keyframe_cond=False,
    )
    server_args = SimpleNamespace(
        attention_backend="fa",
        component_attention_backends={"transformer": "cube_sparse_attn"},
        attention_backend_config={
            "local_cube_size": [4, 4, 4],
            "topk_ratio_list": [0.5],
        },
    )
    assert (
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=1,
            device=torch.device("cpu"),
        )
        is not None
    )

    server_args.attention_backend = "cube_sparse_attn"
    server_args.component_attention_backends["transformer"] = "fa"
    assert (
        _build_cube_attn_metadata(
            server_args,
            packed=packed,
            num_steps=1,
            device=torch.device("cpu"),
        )
        is None
    )


def test_cube_metadata_is_updated_per_step():
    branch = _branch("t2va")
    metadata = SimpleNamespace(current_timestep=-1, topk_ratio_list=[1.0, 0.25])
    seen = []

    def model_forward(_model, _kwargs, step):
        seen.append((step, metadata.current_timestep))
        return (
            torch.zeros(int(branch.update_mask.sum()), 96),
            torch.zeros(branch.audio_pos.numel(), 32),
        )

    minimax_h3_denoise_loop(
        model=SimpleNamespace(prepare_adaln_plans=lambda _: None),
        model_forward=model_forward,
        positive=branch,
        initial_video_rows=torch.zeros(branch.img_pos.numel(), 96),
        initial_audio_rows=torch.zeros(branch.audio_pos.numel(), 32),
        keyframe_cond_rows=None,
        sigmas_video=[1.0, 0.5, 0.0],
        sigmas_audio=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
        attn_metadata=metadata,
    )

    assert seen == [(0, 0), (1, 1)]


def test_native_dit_forward_publishes_cube_metadata_in_forward_context():
    metadata = SimpleNamespace(current_timestep=0, topk_ratio_list=[0.5])
    batch = SimpleNamespace()

    def model(**_kwargs):
        context = get_forward_context()
        assert context.current_timestep == 0
        assert context.attn_metadata is metadata
        assert context.forward_batch is batch
        return torch.zeros(1, 96), torch.zeros(1, 32)

    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    with patch.object(
        MiniMaxH3DenoisingStage,
        "_maybe_get_bcg_runner",
        return_value=None,
    ):
        video, audio = stage._forward_dit(
            model,
            {},
            0,
            batch=batch,
            attn_metadata=metadata,
        )
    assert video.shape == (1, 96)
    assert audio.shape == (1, 32)


def test_grouped_outputs_share_prompt_refinement():
    class Refiner:
        calls = 0

        def refine_prompt_embeds(self, prompt_embeds, refiner_cu, *, device):
            del refiner_cu
            self.calls += 1
            return torch.ones(
                prompt_embeds.shape[0], 5376, dtype=prompt_embeds.dtype, device=device
            )

    model = Refiner()
    conditioning = {}
    first, second = _branch("t2va"), _branch("t2va")

    for branch in (first, second):
        assert _precompute_refined_prompt_embeds(
            model,
            branch,
            device=torch.device("cpu"),
            shared_conditioning=conditioning,
        )

    assert model.calls == 1
    assert first.static_kwargs["prompt_embeds"] is second.static_kwargs["prompt_embeds"]


@pytest.fixture
def single_sp_rank(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.post_training.sp_utils.get_sp_world_size",
        lambda: 1,
    )


def _step_velocities(branch: MiniMaxH3DenoiseBranch, step: int):
    """The (video target, audio) velocities of a DiT that ignores its inputs."""
    generator = torch.Generator().manual_seed(100 + step)
    return (
        torch.randn(int(branch.update_mask.sum()), 96, generator=generator),
        torch.randn(branch.audio_pos.numel(), 32, generator=generator),
    )


def _t2va_rollout_inputs():
    branch = _branch("t2va")
    # A video grid on which the float-form operands differ from the fp32 serving
    # ones; on many grids (shift 12 among them) the two forms agree bit for bit.
    sigmas = {
        name: minimax_h3_time_shift_sigmas(num_steps=4, shift_scale=shift)
        for name, shift in (("video", 2.5), ("audio", 3.0))
    }
    noise = torch.Generator().manual_seed(1)
    initial_video = torch.randn(branch.img_pos.numel(), 96, generator=noise)
    initial_audio = torch.randn(branch.audio_pos.numel(), 32, generator=noise)
    return branch, sigmas, initial_video, initial_audio


def _run_t2va_loop(batch: Req):
    """The denoising stage's rollout calls around the loop, on a t2va layout."""
    branch, sigmas, initial_video, initial_audio = _t2va_rollout_inputs()
    with rollout_scope(batch):
        rollout_plan = prepare_minimax_h3_rollout(
            batch,
            sigmas_video=sigmas["video"],
            sigmas_audio=sigmas["audio"],
            initial_video_rows=initial_video,
            initial_audio_rows=initial_audio,
            device=torch.device("cpu"),
            pipeline_config=_SP_IDENTITY_PIPELINE_CONFIG,
        )
        video_rows, audio_rows = minimax_h3_denoise_loop(
            model=SimpleNamespace(prepare_adaln_plans=lambda _: None),
            model_forward=lambda _model, _kwargs, step: _step_velocities(branch, step),
            positive=branch,
            initial_video_rows=initial_video,
            initial_audio_rows=initial_audio,
            keyframe_cond_rows=None,
            sigmas_video=rollout_plan.sigmas_video,
            sigmas_audio=rollout_plan.sigmas_audio,
            device=torch.device("cpu"),
            video_rollout_step=rollout_plan.video_rollout_step,
            audio_rollout_step=rollout_plan.audio_rollout_step,
        )
        finalize_minimax_h3_rollout(
            batch,
            rollout_plan=rollout_plan,
            video_target_rows=video_rows[branch.video_target_slice],
            audio_target_rows=audio_rows[branch.audio_target_slice],
            pipeline_config=_SP_IDENTITY_PIPELINE_CONFIG,
        )
    return branch, rollout_plan, video_rows, audio_rows


def _rollout_request(**rollout_args) -> Req:
    batch = Req(
        sampling_params=MiniMaxH3SamplingParams(
            prompt="test",
            task="t2va",
            rollout=True,
            rollout_return_dit_trajectory=True,
            **rollout_args,
        )
    )
    batch.generator = [torch.Generator().manual_seed(0)]
    return batch


def _h3_float_form_step(rows, velocity, sigma, sigma_next):
    """The ODE step H3's video rollout has always taken: sigma and the sigma ratio
    rounded from Python doubles to fp32."""
    denoised = rows + torch.tensor(sigma) * velocity
    ratio = torch.tensor(sigma_next / sigma)
    return ratio * rows + (1.0 - ratio) * denoised


def test_video_ode_rollout_keeps_the_h3_float_form_bits(single_sp_rank):
    """miles_diffusion's H3 e2e standards were recorded on the float-form step; the
    serving update and the controller's Euler step land a few ulp away from it."""
    branch, sigmas, initial_video, _ = _t2va_rollout_inputs()
    batch = _rollout_request(rollout_sde_type="ode", rollout_log_prob_no_const=True)
    _, _, video, _ = _run_t2va_loop(batch)

    expected = initial_video[branch.video_target_slice]
    for step in range(len(sigmas["video"]) - 1):
        velocity, _ = _step_velocities(branch, step)
        expected = _h3_float_form_step(
            expected, velocity, sigmas["video"][step], sigmas["video"][step + 1]
        )
    assert torch.equal(video[branch.video_target_slice], expected)


@pytest.mark.parametrize("sde_type", ["cps", "ode"])
def test_stopped_rollout_records_each_step_and_lands_on_x0(single_sp_rank, sde_type):
    """A stop at step i cuts both grids and ends on x_i's x0 prediction, also on the
    ODE path DMD2 takes, whose update reads the cut grid. The recorded x_0..x_i are
    distinct although H3 updates its rows in place, and the loop's DiT times are
    republished as 1 - sigma."""
    stop_step_index = 1
    batch = _rollout_request(
        rollout_extra_streams=(AUDIO_STREAM,),
        rollout_sde_type=sde_type,
        rollout_noise_level=1.0,
        rollout_log_prob_no_const=True,
        rollout_stop_step_index=stop_step_index,
    )
    branch, rollout_plan, video, _ = _run_t2va_loop(batch)

    streams = batch.rollout_trajectory_data.streams
    trajectory = streams[PRIMARY_STREAM].dit_trajectory
    assert trajectory.latents.shape[1] == stop_step_index + 2
    assert streams[AUDIO_STREAM].dit_trajectory.latents.shape[1] == stop_step_index + 2
    for index in range(stop_step_index + 1):
        assert not torch.equal(
            trajectory.latents[0, index], trajectory.latents[0, index + 1]
        )
    x_i = trajectory.latents[0, stop_step_index]
    velocity, _ = _step_velocities(branch, stop_step_index)
    x0_prediction = (
        x_i + torch.tensor(rollout_plan.sigmas_video[stop_step_index]) * velocity
    )
    assert torch.equal(trajectory.latents[0, -1], x0_prediction)
    assert torch.equal(video[branch.video_target_slice], x0_prediction)
    assert torch.equal(
        batch.timesteps,
        torch.tensor([1.0 - sigma for sigma in rollout_plan.sigmas_video[:-1]]),
    )


def test_ode_rollout_audio_is_the_serving_update(single_sp_rank):
    """The audio controller's ODE step is the loop's serving update, so an ODE audio
    rollout leaves the audio rows (and thus the DiT's next input) as serving would."""
    _, _, _, serving_audio = _run_t2va_loop(
        Req(sampling_params=MiniMaxH3SamplingParams(prompt="test"))
    )
    batch = _rollout_request(
        rollout_extra_streams=(AUDIO_STREAM,),
        rollout_sde_type="ode",
        rollout_log_prob_no_const=True,
    )
    branch, _, _, audio = _run_t2va_loop(batch)

    assert torch.equal(audio, serving_audio)
    audio_stream = batch.rollout_trajectory_data.streams[AUDIO_STREAM]
    assert torch.equal(
        audio_stream.dit_trajectory.latents[0, -1], audio[branch.audio_target_slice]
    )


@pytest.mark.parametrize(
    "rollout_args, message",
    [
        ({"task": "fl2va"}, "supports tasks"),
        ({"num_outputs_per_prompt": 2}, "num_outputs_per_prompt=1"),
        (
            {"rollout_noise_level": 0.0, "rollout_log_prob_no_const": True},
            "stream 'primary'.*rollout_noise_level=0",
        ),
        (
            {
                "rollout_extra_streams": (AUDIO_STREAM,),
                "rollout_extra_stream_sde_args": {
                    AUDIO_STREAM: {
                        "rollout_noise_level": 0.0,
                        "rollout_log_prob_no_const": True,
                    }
                },
            },
            "stream 'audio'",
        ),
        # The controller would assert at the first denoising step instead.
        ({"rollout_sde_type": "ode"}, "rollout_log_prob_no_const=True"),
    ],
)
def test_rollout_rejects_what_the_h3_rollout_cannot_run(rollout_args, message):
    rollout_args = {"task": "t2va", **rollout_args}
    with pytest.raises(ValueError, match=message):
        MiniMaxH3SamplingParams(prompt="test", rollout=True, **rollout_args)
