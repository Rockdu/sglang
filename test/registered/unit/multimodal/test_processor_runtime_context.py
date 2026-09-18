"""Multimodal processors accept an explicit runtime without global publication.

    ServerArgs -> RuntimeContext -> registry -> Qwen / GLM / Gemma constructors
                                      |
    PNG history -> load_mm_data -> fast loader -> process_mm_data -> history IDs
                                      |
    history IDs | new placeholder + both PNGs -> process_mm_data_async
                                      |
                     unchanged history + new expansion + plain CPU tensors

The subprocess has no pytest runtime fixtures, model downloads or model weights.
Separate injected contexts retain separate media settings, worker counts and
cache fingerprints, including when the preprocessing cache budget is nonzero.
Original constructors, loading policies and executor lifecycles remain in use;
processors shut down without publishing their config to the global runtime.
"""

import os
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _check_processor_runtime_contexts():
    import asyncio
    import tempfile

    import torch
    from PIL import Image
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        Gemma4AudioFeatureExtractor,
        Gemma4Config,
        Gemma4ImageProcessor,
        Gemma4Processor,
        Gemma4VideoProcessor,
        Glm4vConfig,
        Glm4vImageProcessor,
        Glm4vProcessor,
        Glm4vVideoProcessor,
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
    from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
    from sglang.srt.multimodal.processors.glm4v import (
        Glm4vImageProcessor as SGLangGlm4vProcessor,
    )
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
    from sglang.srt.runtime_context import ParallelContext, RuntimeContext, get_context
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

    def tokenizer(**special_tokens):
        backend = Tokenizer(models.WordLevel(token_ids, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="<|im_end|>",
            additional_special_tokens=tokens[5:],
            extra_special_tokens=special_tokens,
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
    glm_config = Glm4vConfig(
        architectures=["Glm4vForConditionalGeneration"],
        image_token_id=9,
        video_token_id=10,
        image_start_token_id=11,
        image_end_token_id=12,
        video_start_token_id=13,
        video_end_token_id=14,
    )
    glm_processor = Glm4vProcessor(
        tokenizer=tokenizer(),
        image_processor=Glm4vImageProcessor(**vision_kwargs),
        video_processor=Glm4vVideoProcessor(**vision_kwargs),
    )
    gemma_config = Gemma4Config(
        architectures=["Gemma4ForConditionalGeneration"],
        image_token_id=9,
        video_token_id=10,
        audio_token_id=17,
        boi_token_id=15,
        eoi_token_id=16,
        boa_token_id=18,
        eoa_token_id=19,
    )
    gemma_processor = Gemma4Processor(
        tokenizer=tokenizer(
            image_token="<|image|>",
            boi_token="<|image>",
            eoi_token="<image|>",
            audio_token="<|audio|>",
            boa_token="<|audio>",
            eoa_token="<audio|>",
        ),
        image_processor=Gemma4ImageProcessor(max_soft_tokens=70),
        video_processor=Gemma4VideoProcessor(),
        feature_extractor=Gemma4AudioFeatureExtractor(),
    )
    families = [
        (QwenVLImageProcessor, qwen_config, qwen_processor, [7, 5, 8]),
        (SGLangGlm4vProcessor, glm_config, glm_processor, [11, 9, 12]),
        (Gemma4SGLangProcessor, gemma_config, gemma_processor, [9]),
    ]
    global_context = get_context()
    assert not global_context.is_config_namespace_published("mm")

    async def process_images(processor, placeholder, image_paths):
        original_ids = [2] + placeholder + [4]
        loaded = await processor.load_mm_data(
            image_data=image_paths[:1],
        )
        history = (
            processor.process_mm_data(input_ids=original_ids, images=loaded.images)[
                "input_ids"
            ]
            .flatten()
            .tolist()
        )
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

    with tempfile.TemporaryDirectory() as directory:
        image_paths = []
        for index, size in enumerate(((56, 56), (84, 56))):
            path = Path(directory) / f"image_{index}.png"
            Image.new("RGB", size, color=(30 + index, 70, 110)).save(path)
            image_paths.append(str(path))
        for processor_class, config, hf_processor, placeholder in families:
            PROCESSOR_MAPPING.update(
                {model: processor_class for model in processor_class.models}
            )
            args = ServerArgs(
                model_path=config._name_or_path,
                mm_process_config={},
                mm_feature_transport="cpu",
                mm_preprocess_cache_size_mb=1,
                mm_io_worker_num=1,
                mm_processor_worker_num=1,
                disable_fast_image_processor=True,
            )
            context = RuntimeContext(ParallelContext())
            context.set_server_args(args)
            processor = get_mm_processor(
                config, args, hf_processor, transport_mode=None, runtime_context=context
            )
            try:
                assert not global_context.is_config_namespace_published("mm")
                assert isinstance(processor, processor_class)
                assert processor.runtime_context is context
                assert processor.processor_fingerprint is not None
                assert not processor.use_cuda_ipc
                assert processor.mm_io_worker_num == 1
                asyncio.run(process_images(processor, placeholder, image_paths))

                if processor_class is QwenVLImageProcessor:
                    other_args = ServerArgs(
                        model_path=config._name_or_path,
                        mm_process_config={"image": {"do_resize": False}},
                        mm_feature_transport="cpu",
                        mm_preprocess_cache_size_mb=1,
                        mm_io_worker_num=2,
                        mm_processor_worker_num=1,
                        disable_fast_image_processor=True,
                    )
                    other_context = RuntimeContext(ParallelContext())
                    other_context.set_server_args(other_args)
                    other_processor = processor_class(
                        config,
                        other_args,
                        hf_processor,
                        transport_mode=None,
                        runtime_context=other_context,
                    )
                    try:
                        assert other_processor.runtime_context is other_context
                        assert other_processor.processor_fingerprint is not None
                        assert (
                            other_processor.processor_fingerprint
                            != processor.processor_fingerprint
                        )
                        assert other_processor.image_config == {"do_resize": False}
                        assert other_processor.mm_io_worker_num == 2
                        assert processor.image_config == {}
                        assert processor.mm_io_worker_num == 1
                    finally:
                        other_processor.shutdown()
            finally:
                processor.shutdown()
            assert not global_context.is_config_namespace_published("mm")


def test_processors_use_injected_context_without_global_publication():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--check-runtime-contexts"],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if "--check-runtime-contexts" in sys.argv:
        _check_processor_runtime_contexts()
    else:
        import pytest

        raise SystemExit(pytest.main([__file__, "-v"]))
