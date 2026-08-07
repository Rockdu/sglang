"""MiniMax H3 rollout: the two things that would silently corrupt training.

H3 denoises video and audio rows in lockstep inside one packed sequence. RL
samples VIDEO only; audio must still be denoised (the packed sequence couples
them through attention) but is deterministic and untrained.

    loop step k
    ├── video rows  --SDE-->  sampled, log-prob tracked, trained
    └── audio rows  --fused kernel-->  deterministic, recorded as forward input

Two mismatches between H3 and the shared flow-match RL step, both silent:

  1. VELOCITY SIGN
         H3        : x0 = x + sigma*v      =>  x' = x - dt*v
         shared RL : x0 = x - sigma*v      =>  x' = x + dt*v
     so the shared step must be fed -v. Wrong sign still runs, still produces
     plausible latents, and trains on garbage. test_ode_matches_* pins it by
     differencing against H3's own production kernel, with a +v control so the
     assertion cannot pass trivially.

  2. DENOISE STATE SHAPE
     RL noise is sampled at `latents_shape`, which normally comes from
     batch.latents -- but H3 only publishes batch.latents AFTER the loop, and
     steps packed rows [1, N, 96] meanwhile. prepare_rollout takes an explicit
     override; without it the noise buffer would be built from None.
"""

import types
import unittest

import torch

import sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin as rl_mixin_module
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_minimax_h3_euler_ancestral import (
    MiniMaxH3EulerAncestralEta0SchedulerAdapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    _minimax_h3_update_target_rows_,
)
from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
    SchedulerRLMixin,
)

SIGMAS = [1.0, 0.7, 0.4, 0.0]
STEP = 1  # sigma 0.7 -> 0.4, away from both schedule ends
VIDEO_ROWS = (1, 256, 96)


def _batch(sde_type: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        latents=None,  # H3 publishes this only after the denoise loop
        rollout=True,
        rollout_sde_type=sde_type,
        rollout_noise_level=0.7,
        rollout_log_prob_no_const=True,
        rollout_debug_mode=False,
        rollout_sde_step_indices=None,
        _rollout_loop_step_index=STEP,
        _rollout_session_data=None,
    )


def _pipeline_config() -> types.SimpleNamespace:
    # H3 replicates denoise rows across SP ranks; the DiT shards internally.
    return types.SimpleNamespace(shard_latents_for_sp=lambda batch, latents: (latents, False))


def _scheduler() -> MiniMaxH3EulerAncestralEta0SchedulerAdapter:
    return MiniMaxH3EulerAncestralEta0SchedulerAdapter(
        sigmas=torch.tensor(SIGMAS, dtype=torch.float32)
    )


def _h3_fused_step(state: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """H3's production update, run on a copy so the caller's state survives."""
    out = state.clone()
    sigma_curr, sigma_next = SIGMAS[STEP], SIGMAS[STEP + 1]
    ratio = torch.tensor(sigma_next / sigma_curr, dtype=torch.float32)
    _minimax_h3_update_target_rows_(
        out,
        velocity.clone(),
        sigma_t=torch.tensor(sigma_curr, dtype=torch.float32),
        sigma_curr=sigma_curr,
        sigma_ratio=ratio,
        one_minus_sigma_ratio=1.0 - ratio,
        denoised_scratch=torch.empty_like(out),
    )
    return out


class TestMiniMaxH3Rollout(unittest.TestCase):
    def setUp(self):
        # Stay single-process: prepare_rollout consults the SP group size.
        self._orig_get_sp_world_size = rl_mixin_module.get_sp_world_size
        rl_mixin_module.get_sp_world_size = lambda: 1

    def tearDown(self):
        rl_mixin_module.get_sp_world_size = self._orig_get_sp_world_size

    def test_scheduler_satisfies_rollout_gate(self):
        # RolloutDenoisingMixin dispatches on this isinstance check.
        self.assertIsInstance(_scheduler(), SchedulerRLMixin)

    def test_prepare_rollout_uses_explicit_latents_shape(self):
        scheduler = _scheduler()
        batch = _batch("sde")
        scheduler.prepare_rollout(
            batch=batch,
            pipeline_config=_pipeline_config(),
            latents_shape=VIDEO_ROWS,
        )
        session = batch._rollout_session_data
        self.assertEqual(session.latents_shape, VIDEO_ROWS)
        # sigma_max is sigmas[1]: the guard the SDE std uses when sigma == 1.
        self.assertAlmostEqual(session.sigma_max, SIGMAS[1], places=6)

    def test_ode_matches_h3_fused_step_under_negated_velocity(self):
        torch.manual_seed(0)
        state = torch.randn(VIDEO_ROWS, dtype=torch.float32)
        velocity = torch.randn(VIDEO_ROWS, dtype=torch.float32)
        expected = _h3_fused_step(state, velocity)

        scheduler = _scheduler()
        batch = _batch("ode")
        scheduler.prepare_rollout(
            batch=batch,
            pipeline_config=_pipeline_config(),
            latents_shape=VIDEO_ROWS,
        )

        def shared_step(signed_velocity: torch.Tensor) -> torch.Tensor:
            return scheduler.flow_sde_sampling(
                batch,
                model_output=signed_velocity,
                sample=state,
                current_sigma=torch.tensor(SIGMAS[STEP]),
                next_sigma=torch.tensor(SIGMAS[STEP + 1]),
                generator=torch.Generator().manual_seed(0),
            )

        # Algebraically identical, so only fp32 op-order noise should remain.
        self.assertTrue(torch.allclose(shared_step(-velocity), expected, atol=1e-6))
        # Control: the wrong sign must be obviously wrong, not marginally so.
        self.assertGreater((shared_step(velocity) - expected).abs().max().item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
