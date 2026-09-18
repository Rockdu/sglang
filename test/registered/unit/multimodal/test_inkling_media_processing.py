"""Inkling uses the common pipeline with native encoded-media components.

    data/file URL            -> load thread pool -> native encoded media
    encoded image bytes/path -> native image component -> patch views + counts
    encoded audio bytes      -> native feature extractor -> dMel + valid length
                                  |
    two adjacent image pads ------+-> separate source offsets
    expanded history | new pad ---+-> untouched prefix, suffix-only expansion

Real CPU components verify byte/path parity, including escaped file URLs.
The common matcher and builder own
all token matching, offsets and transport; these model hooks only supply media
outputs and replacement fragments. Worker-specific image/audio components must
be called rather than shared components on the owning processor instance.
"""

import asyncio
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

from sglang.srt.multimodal.inkling import (
    InklingAudioFeatureExtractor,
    InklingImageProcessor,
    InklingProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.inkling import InklingMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_processor():
    processor = object.__new__(InklingMultimodalProcessor)
    processor.IMAGE_TOKEN_ID = 101
    processor.AUDIO_TOKEN_ID = 102
    processor.audio_end_token_id = 103
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token_id=101, audio_token_id=102
    )
    processor._tokenizer = None
    processor.image_config = {}
    processor.video_config = {}
    processor.audio_config = {}
    processor.video_preprocessing_device = None
    processor.mm_feature_transport = "cpu"
    processor.use_cuda_ipc = False
    processor.precompute_hash_before_cpu_transfer = False
    processor.inkling_processor = InklingProcessor(
        image_processor=InklingImageProcessor(
            patch_size=4,
            rescale_image_frac=None,
            rescale_image_max_upscaled_long_edge=None,
        ),
        audio_feature_extractor=InklingAudioFeatureExtractor(),
    )
    processor._processor = processor.inkling_processor
    return processor


def test_native_encoded_media_features_and_adjacent_source_offsets(tmp_path):
    processor = _make_processor()
    image_buffer = io.BytesIO()
    Image.new("RGB", (7, 5), (30, 40, 50)).save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()
    image_path = tmp_path / "media with spaces.png"
    image_path.write_bytes(image_bytes)
    image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    audio_bytes = audio_buffer.getvalue()
    audio_url = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode()
    with ThreadPoolExecutor(max_workers=2) as executor:
        processor.io_executor = executor
        loaded = asyncio.run(
            processor.load_mm_data(
                prompt=None,
                image_data=[image_url, image_path.as_uri()],
                audio_data=[audio_url],
            )
        )
    assert loaded.images[0] == image_bytes  # Rust hashes the original encoded bytes.
    image_output = processor.process_images(loaded.images, processor._processor)
    native_image_output = processor.inkling_processor.process_images(
        [image_bytes, str(image_path)]
    )
    torch.testing.assert_close(
        image_output.encoder_inputs["vision_patches_bthwc"],
        native_image_output["vision_patches_bthwc"],
    )
    for item in image_output.items:
        assert (
            item.encoder_inputs["vision_patches_bthwc"].untyped_storage().data_ptr()
            == image_output.encoder_inputs["vision_patches_bthwc"]
            .untyped_storage()
            .data_ptr()
        )

    audio_output = processor.process_audio(loaded.audios, processor._processor)
    native_audio_output = processor.inkling_processor.process_audios([audio_bytes])
    torch.testing.assert_close(
        audio_output.items[0].encoder_inputs["dmel_bins"],
        native_audio_output["dmel_bins"][0],
    )
    media = {"image": image_output, "audio": audio_output}
    original = [80, 101, 101, 81, 102, 82]
    expanded = processor.mm_token_expansion(original, media)
    image_length = image_output.items[0].metadata["num_tokens"]
    assert original == [80, 101, 101, 81, 102, 82]
    output = processor.assemble(original, [image_bytes, str(image_path)], [audio_bytes])
    assert output.input_ids == expanded
    assert output.audio_end_id == 103
    assert [item.offsets for item in output.mm_items] == [
        [(1, image_length)],
        [(1 + image_length, 2 * image_length)],
        [(2 + 2 * image_length, len(expanded) - 2)],
    ]

    next_media = dict(media)
    next_media["image"] = processor.process_images(
        [image_bytes] * 3, processor._processor
    )
    history = expanded
    partial = processor.mm_token_expansion(
        history + [83, 101, 84], next_media, len(history)
    )
    assert partial == history + [83] + [101] * image_length + [84]
    next_output = processor.build_multimodal_inputs(partial, next_media)
    assert [next_output.mm_items[i].offsets for i in (0, 1, 3)] == [
        item.offsets for item in output.mm_items
    ]
    assert next_output.mm_items[2].offsets == [
        (len(history) + 1, len(history) + image_length)
    ]


def test_worker_components_keep_rust_hashes_and_audio_isolated_from_shared_processor():
    processor = _make_processor()
    patches = torch.arange(5 * 12).reshape(5, 1, 2, 2, 3)
    worker_processor = SimpleNamespace(
        process_audios=Mock(
            return_value={"dmel_bins": [torch.ones(2, 3)], "num_audio_tokens": [2]}
        ),
        process_images=Mock(
            return_value={
                "vision_patches_bthwc": patches,
                "num_patches": [2, 3],
                "num_tokens": [2, 3],
                "content_hashes": [11, 22],
            }
        ),
    )
    sources = [b"first", b"second"]
    output = processor.process_images(sources, worker_processor)
    worker_processor.process_images.assert_called_once_with(sources)
    audio_output = processor.process_audio([b"encoded audio"], worker_processor)
    worker_processor.process_audios.assert_called_once_with([b"encoded audio"])
    assert audio_output.items[0].encoder_inputs["dmel_bins"].shape == (2, 3)
    assert [item.metadata["hash"] for item in output.items] == [11, 22]
    torch.testing.assert_close(
        output.items[1].encoder_inputs["vision_patches_bthwc"], patches[2:]
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
