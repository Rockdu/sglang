"""Serving processors keep published settings and share executor lifecycles.

    published ServerArgs -> registry -> Qwen serving processor
    startup opt-in -> token-space route    disabled -> original route
                              |
    PNG history -> load/process -> history IDs + new placeholder + both PNGs
                              |
    process_mm_data_async -> unchanged history + new expansion + CPU tensors

    shared mixin -> one IO pool + one CPU pool + one clone pool -> shutdown
    serving Base -> existing cache -> default / disabled / split budget -> flush

The subprocess uses real HF processors without model downloads or model weights.
"""

import os
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _check_serving_processor_lifecycle():
    import asyncio
    import tempfile
    from contextlib import contextmanager
    from unittest.mock import patch

    import torch
    from PIL import Image
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.managers.multimodal_processor import (
        PROCESSOR_MAPPING,
        get_mm_processor,
    )
    from sglang.srt.multimodal import media_processor
    from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    tokens = [
        "[UNK]",
        "[PAD]",
        "history",
        "current",
        "end",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image|>",
        "<|video|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
        "<|image>",
        "<image|>",
        "<|audio|>",
        "<|audio>",
        "<audio|>",
        "<|im_end|>",
    ]
    token_ids = {token: index for index, token in enumerate(tokens)}

    def tokenizer():
        backend = Tokenizer(models.WordLevel(token_ids, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="<|im_end|>",
            additional_special_tokens=tokens[5:],
        )

    vision_kwargs = {
        "patch_size": 14,
        "size": {"shortest_edge": 56**2, "longest_edge": 112**2},
    }
    qwen_config = Qwen3VLConfig(
        architectures=["Qwen3VLForConditionalGeneration"],
        image_token_id=5,
        video_token_id=6,
        vision_start_token_id=7,
        vision_end_token_id=8,
    )
    qwen_processor = Qwen3VLProcessor(
        tokenizer=tokenizer(),
        image_processor=Qwen2VLImageProcessor(**vision_kwargs),
        video_processor=Qwen3VLVideoProcessor(**vision_kwargs),
    )
    processor_class = QwenVLImageProcessor
    config, hf_processor, placeholder = qwen_config, qwen_processor, [7, 5, 8]
    reset_context()

    @contextmanager
    def check_pool_creation():
        with (
            patch.object(
                media_processor.concurrent.futures,
                "ThreadPoolExecutor",
                wraps=media_processor.concurrent.futures.ThreadPoolExecutor,
            ) as thread_pools,
            patch.object(
                media_processor.concurrent.futures,
                "ProcessPoolExecutor",
                wraps=media_processor.concurrent.futures.ProcessPoolExecutor,
            ) as cpu_pools,
            patch.object(
                media_processor,
                "MultimodalProcessorExecutor",
                wraps=media_processor.MultimodalProcessorExecutor,
            ) as clone_pools,
        ):
            yield
        assert [
            call.kwargs["thread_name_prefix"] for call in thread_pools.call_args_list
        ] == ["sglang-mm-io", "sglang-mm-processor"]
        cpu_pools.assert_called_once()
        clone_pools.assert_called_once()

    async def process_images(processor, placeholder, image_paths):
        original_ids = [2] + placeholder + [4]
        loaded = await processor.load_mm_data(
            image_data=image_paths[:1],
        )
        history = processor.process_mm_data(
            input_ids=original_ids, images=loaded.images
        ).input_ids
        assert len(history) > len(original_ids)
        partial_ids = history + [3] + placeholder + [4]
        boundary = len(history) + 1
        output = await processor.process_mm_data_async(
            image_data=image_paths,
            input_text=partial_ids,
            request_obj=GenerateReqInput(
                input_ids=partial_ids, mm_token_expansion_start_len=boundary
            ),
        )
        assert output.input_ids[:boundary] == partial_ids[:boundary]
        assert len(output.input_ids) > len(partial_ids)
        assert len(output.mm_items) == 2
        for item in output.mm_items:
            assert isinstance(item.feature, torch.Tensor)
            assert item.feature.device.type == "cpu"
            assert item.offsets

    class CacheDefaultProcessor(BaseMultimodalProcessor):
        auto_mm_preprocess_cache_size_mb = 6

        async def process_mm_data_async(self, *args, **kwargs):
            raise NotImplementedError

    # Model default / disabled / explicit budgets retain the same cache lifecycle.
    for budget, expected_mb in ((None, 2), (0, 0), (9, 3)):
        args = ServerArgs(
            model_path="dummy",
            mm_process_config={},
            mm_preprocess_cache_size_mb=budget,
            tokenizer_worker_num=3,
            mm_io_worker_num=1,
            mm_processor_worker_num=1,
        )
        publish(args, role="test")
        cached_processor = CacheDefaultProcessor(
            config, args, hf_processor, None, "unused positional extension"
        )
        try:
            cache = cached_processor.mm_preprocess_cache
            assert cache.max_size_bytes == expected_mb * 1024 * 1024
            assert cache.put("image", b"feature") == bool(expected_mb)
            assert cache.get("image") == (b"feature" if expected_mb else None)
            assert bool(cached_processor.processor_fingerprint) == bool(expected_mb)
        finally:
            cached_processor.shutdown()
        assert len(cache) == 0
        assert cached_processor.io_executor._shutdown
        assert cached_processor.cpu_executor._shutdown_thread

    with tempfile.TemporaryDirectory() as directory:
        image_paths = []
        for index, size in enumerate(((56, 56), (84, 56))):
            path = Path(directory) / f"image_{index}.png"
            Image.new("RGB", size, color=(30 + index, 70, 110)).save(path)
            image_paths.append(str(path))
        PROCESSOR_MAPPING.update(
            {model: processor_class for model in processor_class.models}
        )
        args = ServerArgs(
            model_path="dummy",
            mm_process_config={},
            enable_token_space_processor=True,
            mm_feature_transport="cpu",
            mm_preprocess_cache_size_mb=1,
            mm_io_worker_num=1,
            mm_processor_worker_num=2,
            disable_fast_image_processor=True,
        )
        publish(args, role="test")
        with check_pool_creation():
            processor = get_mm_processor(
                config, args, hf_processor, transport_mode=None
            )
        try:
            assert type(processor) is processor_class.token_space_processor_class
            assert processor.processor_fingerprint is not None
            assert processor.mm_preprocess_cache.put("probe", b"cached")
            processor.clear_preprocess_cache()
            assert processor.mm_preprocess_cache.get("probe") is None
            assert not processor.use_cuda_ipc
            assert processor.mm_io_worker_num == 1
            asyncio.run(process_images(processor, placeholder, image_paths))

            other_args = ServerArgs(
                model_path="dummy",
                mm_process_config={"image": {"do_resize": False}},
                enable_token_space_processor=False,
                mm_feature_transport="cpu",
                mm_preprocess_cache_size_mb=1,
                mm_io_worker_num=2,
                mm_processor_worker_num=2,
                disable_fast_image_processor=True,
            )
            publish(other_args, role="test")
            with check_pool_creation():
                other_processor = processor_class(
                    config,
                    other_args,
                    hf_processor,
                    transport_mode=None,
                )
            try:
                assert not other_processor.use_token_space_processor
                assert other_processor.processor_fingerprint is not None
                assert (
                    other_processor.processor_fingerprint
                    != processor.processor_fingerprint
                )
                assert other_processor.image_config == {"do_resize": False}
                assert other_processor.mm_io_worker_num == 2
                assert processor.image_config == {}
                assert processor.mm_io_worker_num == 1
                legacy = asyncio.run(
                    other_processor.process_mm_data_async(
                        image_data=image_paths[:1],
                        input_text="history <|vision_start|><|image_pad|><|vision_end|> end",
                        request_obj=GenerateReqInput(text=""),
                    )
                )
                assert len(legacy.mm_items) == 1
                assert legacy.mm_items[0].offsets
            finally:
                other_processor.shutdown()
        finally:
            processor.shutdown()
        assert len(processor.mm_preprocess_cache) == 0
        assert processor.io_executor._shutdown
        assert processor.cpu_executor._shutdown_thread
        reset_context()


def test_serving_processors_share_pools_and_preserve_cache_lifecycle():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--check-serving-lifecycle"],
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SGLANG_CPU_WORKERS": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if "--check-serving-lifecycle" in sys.argv:
        _check_serving_processor_lifecycle()
    else:
        import pytest

        raise SystemExit(pytest.main([__file__, "-v"]))
