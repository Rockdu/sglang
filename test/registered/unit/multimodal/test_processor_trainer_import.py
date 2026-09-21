"""A fresh trainer uses shared processor pools without loading inference modules.

fresh Python -> independent Qwen + explicit processor config
                     |
CPU / GPU device -> default: one worker; explicit: requested clone workers
                     |
PNG / video frames -> shared IO + clone pools -> native BatchFeature
                     |
CPU pool + token expansion -> shutdown -> executor submission rejected

Serving Base, model wrappers, ServerArgs and inference kernel imports are forbidden.
"""

import os
import subprocess
import sys
import textwrap

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_trainer_uses_shared_pools_without_inference_imports():
    script = textwrap.dedent(
        """
        import asyncio
        import importlib.abc
        import sys
        import tempfile
        from pathlib import Path

        forbidden = (
            "sglang.kernels",
            "sglang.srt.managers",
            "sglang.srt.model_executor",
            "sglang.srt.model_loader",
            "sglang.srt.models",
            "sglang.srt.layers",
            "sglang.srt.distributed",
            "sglang.srt.server_args",
            "sglang.srt.multimodal.processors.base_processor",
            "sglang.srt.multimodal.processors.qwen_vl",
            "sgl_kernel",
            "flashinfer",
        )

        class RejectInferenceImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                assert not any(
                    fullname == prefix or fullname.startswith(prefix + ".")
                    for prefix in forbidden
                ), fullname

        sys.meta_path.insert(0, RejectInferenceImports())

        from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
        from sglang.srt.multimodal.token_space.qwen_vl import QwenTokenSpaceProcessor

        import torch
        from PIL import Image
        from tokenizers import Tokenizer, models
        from transformers import (
            BatchFeature, PreTrainedTokenizerFast, Qwen2VLImageProcessor,
            Qwen3VLConfig, Qwen3VLProcessor, Qwen3VLVideoProcessor,
        )

        tokens = [
            "[UNK]", "[PAD]", "<|image_pad|>", "<|video_pad|>",
            "<|vision_start|>", "<|vision_end|>",
        ]
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel(
                {token: index for index, token in enumerate(tokens)},
                unk_token="[UNK]",
            )),
            unk_token="[UNK]",
            pad_token="[PAD]",
            additional_special_tokens=tokens[2:],
        )
        hf_processor = Qwen3VLProcessor(
            tokenizer=tokenizer,
            image_processor=Qwen2VLImageProcessor(
                size={"shortest_edge": 56**2, "longest_edge": 56**2},
            ),
            video_processor=Qwen3VLVideoProcessor(
                patch_size=14,
                size={"shortest_edge": 56**2, "longest_edge": 56**2},
            ),
        )
        # Training owns its worker count even when the model declares a serving default.
        worker_cases = [({}, 1)]
        for device in ("cpu", "cuda:0"):
            for requested, expected in ((0, 1), (1, 1), (3, 3)):
                worker_cases.append(({
                    "processor_config": MultimodalProcessorConfig(
                        device=device, mm_processor_worker_num=requested,
                        cpu_worker_num=1,
                    ),
                }, expected))
        for options, expected_workers in worker_cases:
            configured = QwenTokenSpaceProcessor(Qwen3VLConfig(), hf_processor, **options)
            try:
                assert configured.mm_processor_worker_num == expected_workers, options
                executor = configured.mm_processor_executor
                if expected_workers == 1:
                    assert executor is None, options
                else:
                    assert executor._executor._max_workers == expected_workers, options
            finally:
                configured.shutdown()

        processor = QwenTokenSpaceProcessor(
            Qwen3VLConfig(
                image_token_id=2, video_token_id=3,
                vision_start_token_id=4, vision_end_token_id=5,
            ),
            hf_processor,
            processor_config=MultimodalProcessorConfig(
                device="cpu", mm_io_worker_num=2, mm_processor_worker_num=2,
                cpu_worker_num=1, cpu_process_start_method="spawn",
            ),
        )

        async def process_image(image_path):
            loaded = await processor.load_mm_data(image_data=[str(image_path)])
            return await processor.process_media_async(images=loaded.images)

        async def process_video(frames, metadata):
            loaded = await processor.load_mm_data(video_data=[frames])
            return await processor.process_media_async(
                videos=loaded.videos,
                videos_kwargs={
                    "do_sample_frames": False, "video_metadata": [metadata],
                },
            )

        try:
            with tempfile.TemporaryDirectory() as directory:
                image_path = Path(directory) / "image.png"
                image = Image.new("RGB", (56, 56), (17, 39, 81))
                image.save(image_path)
                features = asyncio.run(process_image(image_path))
                expected = hf_processor.image_processor(image, return_tensors="pt")
                assert isinstance(features, BatchFeature)
                assert set(features) == set(expected)
                for name in expected:
                    assert torch.equal(features[name], expected[name]), name
            spec = processor.get_mm_token_expansion_spec(hf_processor, features)
            assert processor.mm_token_expansion([4, 2, 5], spec) == [4, 2, 2, 2, 2, 5]

            frames = (torch.arange(4 * 3 * 56 * 56) % 256).to(torch.uint8)
            frames = frames.reshape(4, 3, 56, 56)
            metadata = {
                "fps": 2.0, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3],
            }
            video_features = asyncio.run(process_video(frames, metadata))
            expected_video = hf_processor(
                text="<|video_pad|>", videos=[frames],
                video_metadata=[dict(metadata)], do_sample_frames=False,
                return_tensors="pt",
            )
            for name in ("pixel_values_videos", "video_grid_thw"):
                assert torch.equal(video_features[name], expected_video[name]), name
            video_spec = processor.get_mm_token_expansion_spec(
                hf_processor, video_features,
            )
            assert processor.mm_token_expansion([3], video_spec) == (
                expected_video["input_ids"][0].tolist()
            )
            assert processor.cpu_executor.submit(sum, [2, 3]).result(timeout=20) == 5
            assert not hasattr(processor, "runtime_context")
            assert not hasattr(processor, "server_args")
            assert not hasattr(processor, "mm_preprocess_cache")
            assert not torch.cuda.is_initialized()
            assert not torch.distributed.is_initialized()
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules for prefix in forbidden
            )
        finally:
            processor.shutdown()
        for executor in (processor.io_executor, processor.cpu_executor):
            try:
                executor.submit(sum, [2, 3])
            except RuntimeError:
                pass
            else:
                raise AssertionError("Processor shutdown left an executor running")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "SGLANG_USE_CPU_ENGINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
