"""Unit tests for the Rollout Image API (serialization, io_struct, rollout_api)."""

import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    RolloutImageRequest,
    RolloutImageResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_serialize,
    base64_to_tensor,
    tensor_to_base64,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDenoisingEnv,
    RolloutDebugTensors,
    RolloutTrajectoryData,
)


# =========================================================================
# serialization.py
# =========================================================================


class TestTensorToBase64Roundtrip(unittest.TestCase):
    """tensor_to_base64 / base64_to_tensor must be lossless."""

    def _roundtrip(self, t: torch.Tensor):
        encoded = tensor_to_base64(t)
        self.assertIsInstance(encoded, str)
        decoded = base64_to_tensor(encoded)
        self.assertTrue(torch.equal(t, decoded), f"Mismatch for shape={t.shape} dtype={t.dtype}")

    def test_float32_1d(self):
        self._roundtrip(torch.randn(16))

    def test_float32_nd(self):
        self._roundtrip(torch.randn(2, 4, 8, 8))

    def test_float16(self):
        self._roundtrip(torch.randn(3, 5).half())

    def test_int64(self):
        self._roundtrip(torch.arange(10))

    def test_bool(self):
        self._roundtrip(torch.tensor([True, False, True]))

    def test_scalar(self):
        self._roundtrip(torch.tensor(3.14))

    def test_empty(self):
        self._roundtrip(torch.empty(0))

    def test_cuda_tensor_moves_to_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        t = torch.randn(4, device="cuda")
        encoded = tensor_to_base64(t)
        decoded = base64_to_tensor(encoded)
        self.assertTrue(torch.equal(t.cpu(), decoded))

    def test_non_contiguous(self):
        t = torch.randn(4, 6)[:, ::2]
        self.assertFalse(t.is_contiguous())
        self._roundtrip(t.contiguous())
        decoded = base64_to_tensor(tensor_to_base64(t))
        self.assertTrue(torch.equal(t.contiguous(), decoded))

    def test_grad_tensor_detaches(self):
        t = torch.randn(3, requires_grad=True)
        encoded = tensor_to_base64(t)
        decoded = base64_to_tensor(encoded)
        self.assertFalse(decoded.requires_grad)
        self.assertTrue(torch.equal(t.detach(), decoded))


class TestTensorFallbackWithoutSafetensors(unittest.TestCase):
    """Test the torch.save/load fallback when safetensors is unavailable."""

    def test_roundtrip_via_torch_fallback(self):
        import importlib
        import sys
        import sglang.multimodal_gen.runtime.entrypoints.post_training.utils as ser_mod

        real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if name == "safetensors.torch":
                raise ImportError("mocked out safetensors")
            return real_import(name, *args, **kwargs)

        t = torch.randn(3, 4)

        with patch("builtins.__import__", side_effect=fake_import):
            encoded = ser_mod.tensor_to_base64(t)
            decoded = ser_mod.base64_to_tensor(encoded)

        self.assertTrue(torch.equal(t, decoded))


class TestMaybeSerialize(unittest.TestCase):
    def test_tensor(self):
        t = torch.randn(2, 3)
        result = _maybe_serialize(t)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["__tensor__"])
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["dtype"], "torch.float32")
        decoded = base64_to_tensor(result["data"])
        self.assertTrue(torch.equal(t, decoded))

    def test_dict_with_tensors(self):
        d = {"a": torch.tensor([1.0]), "b": "hello", "c": 42}
        result = _maybe_serialize(d)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["a"]["__tensor__"])
        self.assertEqual(result["b"], "hello")
        self.assertEqual(result["c"], 42)

    def test_list_with_tensors(self):
        lst = [torch.tensor(1.0), "text", torch.tensor(2.0)]
        result = _maybe_serialize(lst)
        self.assertIsInstance(result, list)
        self.assertTrue(result[0]["__tensor__"])
        self.assertEqual(result[1], "text")
        self.assertTrue(result[2]["__tensor__"])

    def test_nested_structure(self):
        nested = {"level1": {"level2": [torch.tensor(1.0), {"level3": torch.tensor(2.0)}]}}
        result = _maybe_serialize(nested)
        self.assertTrue(result["level1"]["level2"][0]["__tensor__"])
        self.assertTrue(result["level1"]["level2"][1]["level3"]["__tensor__"])

    def test_none_passthrough(self):
        self.assertIsNone(_maybe_serialize(None))

    def test_plain_values_passthrough(self):
        self.assertEqual(_maybe_serialize(42), 42)
        self.assertEqual(_maybe_serialize("hello"), "hello")
        self.assertAlmostEqual(_maybe_serialize(3.14), 3.14)

    def test_tuple_becomes_list(self):
        result = _maybe_serialize((torch.tensor(1.0), 2))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


# =========================================================================
# io_struct.py — RolloutImageRequest / RolloutImageResponse
# =========================================================================


class TestRolloutImageRequest(unittest.TestCase):
    def test_minimal_request(self):
        req = RolloutImageRequest(prompt="a cat")
        self.assertEqual(req.prompt, "a cat")
        self.assertEqual(req.seed, 1024)
        self.assertTrue(req.return_trajectory_latents)
        self.assertFalse(req.rollout_debug_mode)
        self.assertEqual(req.rollout_sde_type, "sde")
        self.assertAlmostEqual(req.rollout_noise_level, 0.7)

    def test_full_request(self):
        req = RolloutImageRequest(
            prompt="test",
            negative_prompt="bad",
            seed=42,
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            true_cfg_scale=3.0,
            rollout_sde_type="cps",
            rollout_noise_level=0.5,
            rollout_log_prob_no_const=True,
            rollout_debug_mode=True,
            return_trajectory_latents=True,
            return_trajectory_decoded=True,
            image_path=["/path/to/img.png"],
            extra_sampling_params={"boundary_ratio": 0.5},
        )
        self.assertEqual(req.rollout_sde_type, "cps")
        self.assertTrue(req.rollout_log_prob_no_const)
        self.assertEqual(req.extra_sampling_params, {"boundary_ratio": 0.5})

    def test_request_json_roundtrip(self):
        req = RolloutImageRequest(prompt="a dog", seed=99)
        data = req.model_dump()
        req2 = RolloutImageRequest(**data)
        self.assertEqual(req, req2)

    def test_request_validation_missing_prompt(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            RolloutImageRequest()


class TestRolloutImageResponse(unittest.TestCase):
    def test_minimal_response(self):
        resp = RolloutImageResponse(
            request_id="test-id",
            prompt="hello",
            seed=42,
        )
        self.assertEqual(resp.request_id, "test-id")
        self.assertIsNone(resp.generated_output)
        self.assertIsNone(resp.trajectory_latents)
        self.assertIsNone(resp.rollout_log_probs)

    def test_response_with_data(self):
        t = torch.randn(2, 3)
        serialized = _maybe_serialize(t)
        resp = RolloutImageResponse(
            request_id="test-id",
            prompt="hello",
            seed=42,
            generated_output=serialized,
            inference_time_s=1.5,
            peak_memory_mb=4096.0,
        )
        self.assertEqual(resp.inference_time_s, 1.5)
        self.assertTrue(resp.generated_output["__tensor__"])

    def test_response_json_roundtrip(self):
        resp = RolloutImageResponse(
            request_id="r1", prompt="p", seed=1,
            rollout_log_probs=_maybe_serialize(torch.tensor([0.5, 0.6])),
        )
        data = resp.model_dump()
        resp2 = RolloutImageResponse(**data)
        self.assertEqual(resp.request_id, resp2.request_id)
        self.assertEqual(resp.rollout_log_probs, resp2.rollout_log_probs)


# =========================================================================
# rollout_api.py — _serialize_rollout_trajectory and _build_response
# =========================================================================

from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import (
    _build_response,
    _serialize_rollout_trajectory,
)


class TestSerializeRolloutTrajectory(unittest.TestCase):
    def test_none_input(self):
        log_probs, debug, env = _serialize_rollout_trajectory(None)
        self.assertIsNone(log_probs)
        self.assertIsNone(debug)
        self.assertIsNone(env)

    def test_log_probs_only(self):
        rtd = RolloutTrajectoryData(
            rollout_log_probs=torch.tensor([-1.0, -2.0]),
        )
        log_probs, debug, env = _serialize_rollout_trajectory(rtd)
        self.assertIsNotNone(log_probs)
        self.assertTrue(log_probs["__tensor__"])
        self.assertIsNone(debug)
        self.assertIsNone(env)

    def test_log_probs_none_in_rtd(self):
        rtd = RolloutTrajectoryData(rollout_log_probs=None)
        log_probs, debug, env = _serialize_rollout_trajectory(rtd)
        self.assertIsNone(log_probs)
        self.assertIsNone(debug)
        self.assertIsNone(env)

    def test_with_debug_tensors(self):
        dt = RolloutDebugTensors(
            rollout_variance_noises=torch.randn(2, 5, 4, 8, 8),
            rollout_prev_sample_means=torch.randn(2, 5, 4, 8, 8),
            rollout_noise_std_devs=torch.randn(2, 5, 1),
            rollout_model_outputs=torch.randn(2, 5, 4, 8, 8),
        )
        rtd = RolloutTrajectoryData(
            rollout_log_probs=torch.tensor([-0.5, -0.6]),
            rollout_debug_tensors=dt,
        )
        log_probs, debug, env = _serialize_rollout_trajectory(rtd)
        self.assertIsNotNone(log_probs)
        self.assertIsNotNone(debug)
        self.assertIsNone(env)
        self.assertIn("rollout_variance_noises", debug)
        self.assertIn("rollout_prev_sample_means", debug)
        self.assertIn("rollout_noise_std_devs", debug)
        self.assertIn("rollout_model_outputs", debug)
        self.assertTrue(debug["rollout_variance_noises"]["__tensor__"])

    def test_debug_tensors_with_none_fields(self):
        dt = RolloutDebugTensors(
            rollout_variance_noises=None,
            rollout_prev_sample_means=torch.randn(1, 2, 4, 4, 4),
            rollout_noise_std_devs=None,
            rollout_model_outputs=None,
        )
        rtd = RolloutTrajectoryData(
            rollout_log_probs=torch.tensor([-0.3]),
            rollout_debug_tensors=dt,
        )
        log_probs, debug, env = _serialize_rollout_trajectory(rtd)
        self.assertIsNotNone(debug)
        self.assertIsNone(debug["rollout_variance_noises"])
        self.assertTrue(debug["rollout_prev_sample_means"]["__tensor__"])

    def test_with_denoising_env(self):
        rtd = RolloutTrajectoryData(
            denoising_env=RolloutDenoisingEnv(
                image_kwargs={"encoder_hidden_states_image": [torch.randn(1, 8)]},
                pos_cond_kwargs={"encoder_hidden_states": torch.randn(1, 8)},
                neg_cond_kwargs={"encoder_hidden_states": torch.randn(1, 8)},
                guidance=torch.tensor([3.5]),
                trajectory_latent_model_inputs=torch.randn(4, 1, 4, 2, 2, 2),
                trajectory_timesteps=torch.tensor([1.0, 0.75, 0.5, 0.25]),
            )
        )
        _, _, env = _serialize_rollout_trajectory(rtd)
        self.assertIsNotNone(env)
        self.assertIn("static", env)
        self.assertIn("trajectory", env)
        self.assertIn("latent_model_inputs", env["trajectory"])
        self.assertIn("timesteps", env["trajectory"])
        self.assertNotIn("noise_preds", env["trajectory"])
        self.assertTrue(env["trajectory"]["latent_model_inputs"]["__tensor__"])


class TestBuildResponse(unittest.TestCase):
    def _make_metrics(self, duration_s: float = 1.0):
        m = types.SimpleNamespace(total_duration_s=duration_s)
        return m

    def test_minimal_output(self):
        batch = OutputBatch(output=[torch.randn(3, 1, 64, 64)])
        batch.metrics = self._make_metrics(2.5)
        resp = _build_response("r1", "prompt", 42, batch)
        self.assertEqual(resp.request_id, "r1")
        self.assertEqual(resp.prompt, "prompt")
        self.assertEqual(resp.seed, 42)
        self.assertIsNotNone(resp.generated_output)
        self.assertIsNone(resp.trajectory_latents)
        self.assertIsNone(resp.rollout_log_probs)
        self.assertAlmostEqual(resp.inference_time_s, 2.5)

    def test_full_response(self):
        batch = OutputBatch(
            output=[torch.randn(3, 1, 64, 64)],
            trajectory_latents=torch.randn(1, 10, 4, 1, 8, 8),
            trajectory_timesteps=torch.linspace(1.0, 0.0, 10),
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.tensor([-0.5]),
            ),
            peak_memory_mb=8192.0,
        )
        batch.metrics = self._make_metrics(5.0)
        resp = _build_response("r2", "test", 99, batch)
        self.assertIsNotNone(resp.trajectory_latents)
        self.assertIsNotNone(resp.trajectory_timesteps)
        self.assertIsNotNone(resp.rollout_log_probs)
        self.assertIsNone(resp.rollout_debug_tensors)
        self.assertAlmostEqual(resp.peak_memory_mb, 8192.0)

    def test_no_metrics(self):
        batch = OutputBatch(output=[torch.randn(3, 1, 64, 64)])
        batch.metrics = None
        resp = _build_response("r3", "p", 1, batch)
        self.assertIsNone(resp.inference_time_s)

    def test_zero_metrics(self):
        batch = OutputBatch(output=[torch.randn(3, 1, 64, 64)])
        batch.metrics = self._make_metrics(0.0)
        resp = _build_response("r4", "p", 1, batch)
        self.assertIsNone(resp.inference_time_s)

    def test_none_output(self):
        batch = OutputBatch(output=None)
        batch.metrics = None
        resp = _build_response("r5", "p", 1, batch)
        self.assertIsNone(resp.generated_output)

    def test_zero_peak_memory(self):
        batch = OutputBatch(output=None, peak_memory_mb=0.0)
        batch.metrics = None
        resp = _build_response("r6", "p", 1, batch)
        self.assertIsNone(resp.peak_memory_mb)


# =========================================================================
# rollout_api.py — endpoint (async, with mocks)
# =========================================================================


class TestRolloutImagesEndpoint(unittest.IsolatedAsyncioTestCase):
    """Test the rollout_images endpoint with mocked dependencies."""

    def _make_output_batch(self, **overrides):
        defaults = dict(
            output=[torch.randn(3, 1, 64, 64)],
            trajectory_latents=torch.randn(1, 5, 4, 1, 8, 8),
            trajectory_timesteps=torch.linspace(1.0, 0.0, 5),
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.tensor([-1.0]),
            ),
            peak_memory_mb=1024.0,
        )
        defaults.update(overrides)
        batch = OutputBatch(**defaults)
        batch.metrics = types.SimpleNamespace(total_duration_s=1.0)
        return batch

    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.async_scheduler_client")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.get_global_server_args")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.build_sampling_params")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.prepare_request")
    async def test_success_path(self, mock_prepare, mock_build_sp, mock_get_args, mock_client):
        from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import rollout_images

        mock_get_args.return_value = MagicMock()
        mock_build_sp.return_value = MagicMock()
        mock_prepare.return_value = MagicMock()
        output_batch = self._make_output_batch()
        mock_client.forward = AsyncMock(return_value=output_batch)

        request = RolloutImageRequest(prompt="a cat", seed=42)
        response = await rollout_images(request)

        self.assertEqual(response.status_code, 200)
        mock_client.forward.assert_awaited_once()

        # Check that build_sampling_params was called with rollout=True
        call_kwargs = mock_build_sp.call_args
        self.assertTrue(call_kwargs[1].get("rollout") or call_kwargs.kwargs.get("rollout"))

    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.async_scheduler_client")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.get_global_server_args")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.build_sampling_params")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.prepare_request")
    async def test_extra_sampling_params_merged(self, mock_prepare, mock_build_sp, mock_get_args, mock_client):
        from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import rollout_images

        mock_get_args.return_value = MagicMock()
        mock_build_sp.return_value = MagicMock()
        mock_prepare.return_value = MagicMock()
        mock_client.forward = AsyncMock(return_value=self._make_output_batch())

        request = RolloutImageRequest(
            prompt="test",
            extra_sampling_params={"boundary_ratio": 0.5, "num_frames": 1},
        )
        await rollout_images(request)

        call_kwargs = mock_build_sp.call_args[1]
        self.assertAlmostEqual(call_kwargs["boundary_ratio"], 0.5)
        self.assertEqual(call_kwargs["num_frames"], 1)

    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.async_scheduler_client")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.get_global_server_args")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.build_sampling_params")
    async def test_invalid_sampling_params_returns_400(self, mock_build_sp, mock_get_args, mock_client):
        from fastapi import HTTPException
        from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import rollout_images

        mock_get_args.return_value = MagicMock()
        mock_build_sp.side_effect = ValueError("bad param")

        request = RolloutImageRequest(prompt="test")
        with self.assertRaises(HTTPException) as ctx:
            await rollout_images(request)
        self.assertEqual(ctx.exception.status_code, 400)

    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.async_scheduler_client")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.get_global_server_args")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.build_sampling_params")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.prepare_request")
    async def test_scheduler_error_returns_500(self, mock_prepare, mock_build_sp, mock_get_args, mock_client):
        from fastapi import HTTPException
        from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import rollout_images

        mock_get_args.return_value = MagicMock()
        mock_build_sp.return_value = MagicMock()
        mock_prepare.return_value = MagicMock()
        mock_client.forward = AsyncMock(side_effect=RuntimeError("boom"))

        request = RolloutImageRequest(prompt="test")
        with self.assertRaises(HTTPException) as ctx:
            await rollout_images(request)
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.async_scheduler_client")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.get_global_server_args")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.build_sampling_params")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.prepare_request")
    async def test_result_error_returns_500(self, mock_prepare, mock_build_sp, mock_get_args, mock_client):
        from fastapi import HTTPException
        from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import rollout_images

        mock_get_args.return_value = MagicMock()
        mock_build_sp.return_value = MagicMock()
        mock_prepare.return_value = MagicMock()
        error_batch = OutputBatch(error="model exploded")
        error_batch.metrics = None
        mock_client.forward = AsyncMock(return_value=error_batch)

        request = RolloutImageRequest(prompt="test")
        with self.assertRaises(HTTPException) as ctx:
            await rollout_images(request)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("model exploded", ctx.exception.detail)

    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.async_scheduler_client")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.get_global_server_args")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.build_sampling_params")
    @patch("sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api.prepare_request")
    async def test_none_values_filtered_from_sampling_kwargs(self, mock_prepare, mock_build_sp, mock_get_args, mock_client):
        from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import rollout_images

        mock_get_args.return_value = MagicMock()
        mock_build_sp.return_value = MagicMock()
        mock_prepare.return_value = MagicMock()
        mock_client.forward = AsyncMock(return_value=self._make_output_batch())

        request = RolloutImageRequest(prompt="test", width=None, guidance_scale=None)
        await rollout_images(request)

        call_kwargs = mock_build_sp.call_args[1]
        self.assertNotIn("width", call_kwargs)
        self.assertNotIn("guidance_scale", call_kwargs)
        self.assertIn("rollout", call_kwargs)


if __name__ == "__main__":
    unittest.main()
