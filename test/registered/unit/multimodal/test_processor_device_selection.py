"""Serving resolves the current runtime policy; training uses explicit configuration.

    ServerArgs.base_gpu_id + current runtime policy -> serving device
    Explicit processor configuration -------------> training device
      Serving device policy --------->|-> image kwargs <- model video override
                                      |-> video kwargs <- request / model override
    CPU transport -> temporary CUDA pool -> CPU features -> release the pool

Regression: the device decision read the published global ServerArgs, so every
processor answered with one process-wide device. The encode-server DP workers
each drive their own GPU, which no process-global value can express — the
device has to come from what the worker was handed.
Video-only and mixed requests use the same default device; PIL keeps it unset.
The model video override also controls images in a mixed request.
Captured modality kwargs occupy distinct keys in the merged media output.
Runtime policy changes affect serving device/competition checks on existing instances.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from transformers.processing_utils import ProcessorMixin

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
from sglang.srt.multimodal.processors.token_space_processor import (
    TokenSpaceMultimodalProcessor,
)
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BASE = "sglang.srt.multimodal.processors.base_processor"


class _Processor:
    pass


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _make(**fields):
    """Both surfaces, because the device decision reads both.

    `base_gpu_id` is the instance's own -- two engines in one process keep
    different ones, which `test_publishing_another_config_does_not_move_the_device`
    pins -- so it stays on the record the processor holds. `rl_on_policy_target`
    is the process's, so it is published.
    """
    server_args = ServerArgs(model_path="dummy", **fields)
    publish(server_args, role="tokenizer")
    processor = _StubProcessor.__new__(_StubProcessor)
    processor.server_args = server_args
    return processor


class TestFastImageProcessorDevice(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _device(self, processor, **platform):
        flags = {"_is_cpu": False, "_is_xpu": False, "_is_npu": False}
        flags.update(platform)
        with patch.multiple(BASE, **flags):
            return processor._fast_image_processor_device(_Processor())

    def test_device_follows_the_instance_base_gpu_id(self):
        self.assertEqual(self._device(_make(base_gpu_id=3)), "cuda:3")

    def test_engines_in_one_process_keep_their_own_device(self):
        first, second = _make(base_gpu_id=0), _make(base_gpu_id=5)
        self.assertEqual(self._device(first), "cuda:0")
        self.assertEqual(self._device(second), "cuda:5")

    def test_publishing_another_config_does_not_move_the_device(self):
        processor = _make(base_gpu_id=2)
        override = get_context().override_server_args(base_gpu_id=7)
        override.install()
        self.addCleanup(override.restore)
        self.assertEqual(self._device(processor), "cuda:2")

    def test_rl_on_policy_target_forces_cpu(self):
        processor = _make(base_gpu_id=3, rl_on_policy_target="fsdp")
        self.assertEqual(self._device(processor), "cpu")

    def test_runtime_policy_changes_update_existing_serving_processor(self):
        class ImageProcessor:
            pass

        processor = _make(base_gpu_id=3, mm_process_config={})
        with patch.multiple(
            BASE,
            _is_cpu=False,
            _is_xpu=False,
            _is_npu=False,
            BaseImageProcessor=ImageProcessor,
        ):
            processor = _StubProcessor(
                None, processor.server_args, _Processor(), None, skip_mm_pool=True
            )
            self.addCleanup(processor.shutdown)
            processor._processor = SimpleNamespace(image_processor=ImageProcessor())
            processor.disable_fast_image_processor = False
            for target, expected_device, competes in (
                (None, "cuda:3", True),
                ("fsdp", "cpu", False),
                (None, "cuda:3", True),
            ):
                with self.subTest(target=target):
                    override = get_context().override_server_args(
                        rl_on_policy_target=target
                    )
                    override.install()
                    try:
                        self.assertEqual(
                            processor._fast_image_processor_device(_Processor()),
                            expected_device,
                        )
                        self.assertEqual(
                            processor._preprocessing_competes_with_the_scheduler(),
                            competes,
                        )
                    finally:
                        override.restore()

    def test_cpu_and_xpu_platforms_win_over_base_gpu_id(self):
        processor = _make(base_gpu_id=3)
        self.assertEqual(self._device(processor, _is_cpu=True), "cpu")
        self.assertEqual(self._device(processor, _is_xpu=True), "xpu")

    def test_npu_glm4v_leaves_the_device_unset(self):
        class Glm4vProcessor:
            pass

        processor = _make(base_gpu_id=3)
        with patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=True):
            device = processor._fast_image_processor_device(Glm4vProcessor())
        self.assertIsNone(device)

    def test_auto_workers_follow_each_instances_preprocessing_device(self):
        class ImageProcessor:
            pass

        cases = (
            ("cpu", False, None, 2),
            ("cuda", False, None, 1),
            ("cuda", True, None, 2),
            ("cpu", False, 5, 5),
            ("cuda", False, 5, 1),
        )
        with patch(f"{BASE}.BaseImageProcessor", ImageProcessor):
            for platform, use_pil, declared_workers, expected in cases:
                with self.subTest(platform=platform, use_pil=use_pil):
                    processor = _make()
                    processor._processor = SimpleNamespace(
                        image_processor=ImageProcessor()
                    )
                    processor.disable_fast_image_processor = use_pil
                    processor.auto_mm_processor_worker_num = declared_workers
                    with patch.multiple(
                        BASE, _is_cpu=platform == "cpu", _is_xpu=False, _is_npu=False
                    ):
                        self.assertEqual(
                            processor._resolve_auto_mm_processor_worker_num(), expected
                        )

    def test_media_dispatch_keeps_video_device_and_override_priority(self):
        def capture_options(sources, processor, **kwargs):
            return {f"{sources[0]}_options": kwargs}

        # CPU-only capture at the component boundary catches a missing video device.
        cases = (
            (None, False, None, None, "cuda:3"),
            (["image"], False, None, None, "cuda:3"),
            (None, False, "cpu", None, "cpu"),
            (None, False, "cuda:5", "cpu", "cpu"),
            (["image"], False, "cuda:5", "cpu", "cpu"),
            (None, True, None, None, None),
        )
        for images, use_pil, requested_device, model_device, expected_device in cases:
            with self.subTest(
                images=images,
                use_pil=use_pil,
                requested_device=requested_device,
                model_device=model_device,
            ):
                processor = TokenSpaceMultimodalProcessor(
                    None,
                    _Processor(),
                    processor_config=MultimodalProcessorConfig(
                        device="cuda:3",
                        mm_processor_worker_num=1,
                    ),
                )
                processor.shutdown()
                processor.use_token_space_processor = True
                processor._processor = object.__new__(ProcessorMixin)
                processor._tokenizer = SimpleNamespace(init_kwargs={})
                processor._processor.tokenizer = processor._tokenizer
                processor.image_config = {}
                processor.video_config = {}
                processor.audio_config = {}
                processor.disable_fast_image_processor = use_pil
                processor.video_preprocessing_device = model_device
                videos_kwargs = (
                    {} if requested_device is None else {"device": requested_device}
                )
                with (
                    patch.object(
                        processor,
                        "_temporary_fast_processor_cuda_pool",
                        return_value=nullcontext(),
                    ),
                    patch.object(
                        processor,
                        "process_images",
                        new=capture_options,
                    ),
                    patch.object(
                        processor,
                        "process_videos",
                        new=capture_options,
                    ),
                ):
                    output = processor.process_media(
                        images=images,
                        videos=["video"],
                        videos_kwargs=videos_kwargs,
                    )
                self.assertEqual(output["video_options"].get("device"), expected_device)
                if images:
                    self.assertEqual(output["image_options"]["device"], expected_device)


class TestFastImageProcessorMemoryPool(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _processor(self, *, transport="cpu", precompute_hash=False):
        processor = _make(base_gpu_id=0)
        processor.mm_feature_transport = transport
        processor.use_token_space_processor = False
        processor.precompute_hash_before_cpu_transfer = precompute_hash
        return processor

    def test_pool_is_limited_to_immediate_cpu_transport(self):
        cases = (
            (self._processor(), "cuda:0", True),
            (self._processor(transport="cuda_ipc"), "cuda:0", False),
            (self._processor(transport="cuda_vmm"), "cuda:0", False),
            (self._processor(precompute_hash=True), "cuda:0", False),
            (self._processor(), "cpu", False),
            (self._processor(), None, False),
        )
        for processor, device, expected in cases:
            with (
                self.subTest(device=device, transport=processor.mm_feature_transport),
                patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
                patch(f"{BASE}.torch.cuda.MemPool", return_value="pool") as mem_pool,
                patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=nullcontext()),
            ):
                with processor._temporary_fast_processor_cuda_pool(device):
                    pass
                self.assertEqual(mem_pool.called, expected)

    def test_processor_call_uses_private_pool_until_cpu_copy_finishes(self):
        class ImageProcessor:
            pass

        class Feature:
            def to(self, device):
                events.append(("copy", device))

        feature = Feature()

        class Processor:
            image_processor = ImageProcessor()
            tokenizer = SimpleNamespace(bos_token=None)

            def __call__(self, **kwargs):
                events.append(("call", kwargs["device"]))
                return {"pixel_values": feature}

        events = []
        processor = self._processor()
        processor._processor = Processor()
        processor._tokenizer = processor._processor.tokenizer
        processor._tokenizer_auto_adds_specials = False
        processor.disable_fast_image_processor = False
        processor.image_config = {}
        processor.video_config = {}
        processor.audio_config = {}
        processor.FEATURE_NAMES = ["pixel_values"]

        class PoolContext:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("exit")

        with (
            patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=False),
            patch(f"{BASE}.BaseImageProcessor", ImageProcessor),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool"),
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=PoolContext()),
            patch(f"{BASE}.torch.Tensor", Feature),
        ):
            processor.process_mm_data("test", images=["image"])

        self.assertEqual(
            events,
            ["enter", ("call", "cuda:0"), ("copy", "cpu"), "exit"],
        )


if __name__ == "__main__":
    unittest.main()
