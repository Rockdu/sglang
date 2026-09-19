"""Gemma4 media processing and token expansion use independent CPU boundaries.

    decoded image   -> image component -> pixels + local positions + token count
    video reader    -> sample/decode   -> video component + original timestamps
    waveform        -> aligned padding -> audio component + valid-frame mask
                         |
                    ProcessedMedia (reusable; no prompt IDs)
                         |
    IDs + boundary ------+---> expansion ---> final IDs -> serving offsets
    old expanded history | new image  ---> unchanged history + new image block

    per-source preprocessing settings -> native training tensors
        images: zero/-1 patch padding   -> original patch-width serving views
        videos: real frames only       -> original frame/patch-width serving views
        audios: feature/mask padding    -> original time-length serving views

Both Gemma4 variants use real HF components for media parity and lightweight
vision components for source offsets. Serving views share training storage.
Video readers close after sampling; frame lists and tensor inputs preserve caller
timelines independently of the token prefix; per-source timelines override batch metadata.
"""

from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    Gemma4AudioFeatureExtractor,
    Gemma4ImageProcessor,
    Gemma4Processor,
    Gemma4UnifiedAudioFeatureExtractor,
    Gemma4UnifiedImageProcessor,
    Gemma4UnifiedProcessor,
    Gemma4UnifiedVideoProcessor,
    Gemma4VideoProcessor,
)
from transformers.video_utils import VideoMetadata

from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
from sglang.srt.multimodal.processors.gemma4_unified import Gemma4UnifiedSGLangProcessor
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _Tokenizer:
    init_kwargs = {}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        assert "<" not in text
        return [1000 + ord(char) for char in text]

    def decode(self, *args, **kwargs):
        raise AssertionError("The original token IDs must not be decoded")

    def __call__(self, *args, **kwargs):
        raise AssertionError("The full prompt must not be retokenized")


class _ImageProcessor:
    max_soft_tokens = 99

    def __call__(self, images, **kwargs):
        return {
            "pixel_values": torch.ones(len(images), 4, 12),
            "image_position_ids": torch.zeros(len(images), 4, 2, dtype=torch.int64),
            "num_soft_tokens_per_image": [
                min(
                    image.width // 4,
                    kwargs.get("max_soft_tokens", self.max_soft_tokens),
                )
                for image in images
            ],
        }


class _VideoProcessor:
    num_frames = 2

    def __init__(self):
        self.calls = []

    def __call__(self, videos, **kwargs):
        self.calls.append(kwargs)
        return {
            "pixel_values_videos": torch.ones(len(videos), 2, 4, 12),
            "video_position_ids": torch.zeros(len(videos), 2, 4, 2, dtype=torch.int64),
            "num_soft_tokens_per_video": [2] * len(videos),
            "video_metadata": [
                VideoMetadata(**item) for item in kwargs["video_metadata"]
            ],
        }


def _make_processor(unified=False):
    processor_class = Gemma4UnifiedSGLangProcessor if unified else Gemma4SGLangProcessor
    hf_processor_class = Gemma4UnifiedProcessor if unified else Gemma4Processor
    processor = object.__new__(processor_class)
    hf_processor = object.__new__(hf_processor_class)
    hf_processor.tokenizer = _Tokenizer()
    hf_processor.image_processor = _ImageProcessor()
    hf_processor.video_processor = _VideoProcessor()
    hf_processor.feature_extractor = (
        Gemma4UnifiedAudioFeatureExtractor()
        if unified
        else Gemma4AudioFeatureExtractor()
    )
    processor._processor = hf_processor
    processor._tokenizer = hf_processor.tokenizer
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token_id=12, video_token_id=13, audio_token_id=22
    )
    processor.IM_START_TOKEN_ID = 10
    processor.IM_END_TOKEN_ID = 11
    processor.AUDIO_START_TOKEN_ID = 20
    processor.AUDIO_END_TOKEN_ID = 21
    processor.image_config = {}
    processor.video_config = {}
    processor.audio_config = {"sampling_rate": 16000}
    processor.video_preprocessing_device = None
    processor.mm_feature_transport = "cpu"
    processor.disable_fast_image_processor = True
    processor.use_cuda_ipc = False
    processor.precompute_hash_before_cpu_transfer = False
    return processor


@pytest.mark.parametrize("unified", [False, True], ids=["gemma4", "unified"])
def test_media_outputs_keep_source_views_and_expand_only_the_new_suffix(unified):
    processor = _make_processor(unified)
    image0, image1 = Image.new("RGB", (8, 4)), Image.new("RGB", (12, 4))
    media = processor.process_media(
        images=[image0, image1],
        videos=[torch.zeros(2, 3, 4, 4, dtype=torch.uint8)],
        audios=[np.zeros(641, dtype=np.float32)],
    )
    assert set(media) == {"image", "video", "audio"}
    image_output = media["image"]
    pixels = image_output.encoder_inputs["pixel_values"]
    for index, item in enumerate(image_output.items):
        assert item.media_id == ("image", index)
        assert (
            item.encoder_inputs["pixel_values"].untyped_storage().data_ptr()
            == pixels.untyped_storage().data_ptr()
        )
        assert item.encoder_inputs["image_position_ids"].shape == (1, 4, 2)
    assert "input_ids" not in media["audio"].encoder_inputs
    assert "input_features_mask" in media["audio"].encoder_inputs
    assert media["video"].items[0].metadata["timestamps"] == [0, 1 / 24]

    # Two adjacent equal-pad images must remain two independently owned sources.
    original = [903, 12, 12, 904, 13, 905, 22, 906]
    expanded = processor.mm_token_expansion(original, media)
    assert original == [903, 12, 12, 904, 13, 905, 22, 906]
    output = processor.build_multimodal_inputs(expanded, media)
    assert output.input_ids == expanded
    assert output.mm_items[0].offsets == [(2, 3)]
    assert output.mm_items[1].offsets == [(6, 8)]
    assert len(output.mm_items[2].offsets) == 2
    assert output.mm_items[0].image_position_ids.shape == (1, 4, 2)
    assert output.mm_items[2].video_position_ids.shape == (1, 2, 4, 2)
    assert output.mm_items[3].input_features_mask.shape[0] == 1
    training = processor.build_multimodal_inputs(expanded, media, consumer="training")
    assert training["pixel_values"] is pixels
    assert (
        training["input_features_mask"]
        is media["audio"].encoder_inputs["input_features_mask"]
    )

    history = expanded
    next_media = dict(media)
    next_media["image"] = processor.process_images(
        [image0, image1, Image.new("RGB", (16, 4))], processor._processor
    )
    partial = processor.mm_token_expansion(
        history + [907, 12, 908], next_media, len(history)
    )
    assert partial == history + [907, 10, 12, 12, 12, 12, 11, 908]
    next_output = processor.build_multimodal_inputs(partial, next_media)
    assert [next_output.mm_items[i].offsets for i in (0, 1, 3, 4)] == [
        item.offsets for item in output.mm_items
    ]
    assert next_output.mm_items[2].offsets == [(len(history) + 2, len(history) + 5)]
    assert media["image"].items[0].metadata == {"num_soft_tokens": 2}


@pytest.mark.parametrize("unified", [False, True], ids=["gemma4", "unified"])
def test_video_sampling_stays_inside_process_and_retains_original_timeline(unified):
    processor = _make_processor(unified)
    decoder = object.__new__(VideoDecoderWrapper)
    decoder.get_frames_at = Mock(return_value=np.zeros((2, 4, 4, 3), dtype=np.uint8))
    decoder.close = Mock()
    with (
        patch.object(VideoDecoderWrapper, "__len__", return_value=6),
        patch.object(
            VideoDecoderWrapper, "avg_fps", new_callable=PropertyMock, return_value=0.5
        ),
    ):
        result = processor.process_videos([decoder], processor._processor)
    decoder.get_frames_at.assert_called_once_with([0, 3])
    decoder.close.assert_called_once()
    assert result.items[0].metadata["timestamps"] == [0, 6]
    assert result.items[0].encoder_inputs["video_position_ids"].shape == (1, 2, 4, 2)
    kwargs = processor._processor.video_processor.calls[0]
    assert kwargs["do_sample_frames"] is False
    assert kwargs["return_metadata"] is True
    assert kwargs["video_metadata"][0]["frames_indices"] == [0, 3]


@pytest.mark.parametrize("unified", [False, True], ids=["gemma4", "unified"])
def test_audio_padding_matches_real_component_and_preserves_local_masks(unified):
    processor = _make_processor(unified)
    waveform = np.linspace(-0.1, 0.1, 641, dtype=np.float32)
    options = {
        "sampling_rate": 16000,
        "padding": True,
        "return_tensors": "pt",
        "truncation": False,
    }
    result = processor.process_audio([waveform], processor._processor, **options)
    multiple = processor._get_audio_pad_multiple(processor._processor)
    padded = np.pad(waveform, (0, (-len(waveform)) % multiple))
    expected = processor._processor.feature_extractor([padded], **options)
    for key, tensor in expected.items():
        torch.testing.assert_close(result.encoder_inputs[key], tensor)
    mask = expected["input_features_mask"][0]
    count = int(mask.sum()) if unified else int(mask[::4].sum())
    assert result.items[0].metadata["num_audio_tokens"] == count
    assert result.items[0].encoder_inputs["input_features_mask"].shape[0] == 1


@pytest.mark.parametrize("unified", [False, True], ids=["gemma4", "unified"])
def test_vision_features_match_real_components(unified):
    processor = _make_processor(unified)
    image_cls = Gemma4UnifiedImageProcessor if unified else Gemma4ImageProcessor
    video_cls = Gemma4UnifiedVideoProcessor if unified else Gemma4VideoProcessor
    processor._processor.image_processor = image_cls(max_soft_tokens=70)
    processor._processor.video_processor = video_cls(max_soft_tokens=70)
    images = [Image.new("RGB", (96, 64), (30, 40, 50))]
    videos = [
        torch.arange(2 * 3 * 96 * 96, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(2, 3, 96, 96)
    ]
    metadata = [{"fps": 2, "total_num_frames": 8, "frames_indices": [1, 3]}]
    image_output = processor.process_images(
        images, processor._processor, return_tensors="pt"
    )
    video_output = processor.process_videos(
        videos, processor._processor, video_metadata=metadata, return_tensors="pt"
    )
    expected_images = processor._processor.image_processor(images, return_tensors="pt")
    expected_videos = processor._processor.video_processor(
        videos,
        video_metadata=metadata,
        do_sample_frames=False,
        return_tensors="pt",
        return_metadata=True,
    )
    for actual, expected in (
        (image_output, expected_images),
        (video_output, expected_videos),
    ):
        for key, tensor in actual.encoder_inputs.items():
            torch.testing.assert_close(tensor, expected[key])
    assert (
        image_output.items[0].metadata["num_soft_tokens"]
        == expected_images["num_soft_tokens_per_image"][0]
    )
    assert (
        video_output.items[0].metadata["num_soft_tokens"]
        == expected_videos["num_soft_tokens_per_video"][0]
    )
    assert video_output.items[0].metadata["timestamps"] == [0.5, 1.5]
    supplied = VideoMetadata(**metadata[0])
    result = processor.process_videos(
        videos, processor._processor, video_metadata=[supplied], return_tensors="pt"
    )
    assert result.items[0].metadata["timestamps"] == [0.5, 1.5]
    supplied.frames_indices[0] = 0
    assert result.items[0].metadata["timestamps"] == [0.5, 1.5]


@pytest.mark.parametrize("unified", [False, True], ids=["gemma4", "unified"])
def test_per_source_image_budgets_preserve_native_patches_and_shared_views(unified):
    processor = _make_processor(unified)
    image_cls = Gemma4UnifiedImageProcessor if unified else Gemma4ImageProcessor
    processor._processor.image_processor = image_cls(max_soft_tokens=70)
    source = Image.new("RGB", (96, 64), (30, 40, 50))
    history = processor.process_media(images=[source])
    next_media = processor.process_media(
        images=[
            {"url": source, "preprocess_kwargs": {"max_soft_tokens": 70}},
            source,
        ],
        images_kwargs={"max_soft_tokens": 280},
    )
    old_output = history["image"]
    output = next_media["image"]
    assert output.items[0].metadata == old_output.items[0].metadata
    assert (
        output.items[1].metadata["num_soft_tokens"]
        > output.items[0].metadata["num_soft_tokens"]
    )
    assert [item.media_id for item in output.items] == [("image", 0), ("image", 1)]
    pixels = output.encoder_inputs["pixel_values"]
    positions = output.encoder_inputs["image_position_ids"]
    assert isinstance(pixels, torch.Tensor) and isinstance(positions, torch.Tensor)
    history_patches = old_output.encoder_inputs["pixel_values"].shape[1]
    torch.testing.assert_close(
        pixels[0, :history_patches], old_output.encoder_inputs["pixel_values"][0]
    )
    torch.testing.assert_close(
        positions[0, :history_patches],
        old_output.encoder_inputs["image_position_ids"][0],
    )
    assert not pixels[0, history_patches:].any()
    assert positions[0, history_patches:].eq(-1).all()
    torch.testing.assert_close(
        output.items[0].encoder_inputs["pixel_values"],
        old_output.items[0].encoder_inputs["pixel_values"],
    )
    torch.testing.assert_close(
        output.items[0].encoder_inputs["image_position_ids"],
        old_output.items[0].encoder_inputs["image_position_ids"],
    )
    assert all(
        item.encoder_inputs["pixel_values"].untyped_storage().data_ptr()
        == pixels.untyped_storage().data_ptr()
        for item in output.items
    )
    expanded = processor.mm_token_expansion([12, 12], next_media)
    training = processor.build_multimodal_inputs(
        expanded, next_media, consumer="training"
    )
    assert training["pixel_values"] is pixels


@pytest.mark.parametrize("unified", [False, True], ids=["gemma4", "unified"])
def test_different_video_lengths_pack_only_real_frames_with_shared_tensor_views(
    unified,
):
    processor = _make_processor(unified)
    video_cls = Gemma4UnifiedVideoProcessor if unified else Gemma4VideoProcessor
    processor._processor.video_processor = video_cls(max_soft_tokens=70)
    first = torch.zeros(2, 3, 96, 96, dtype=torch.uint8)
    second = torch.ones(1, 3, 96, 96, dtype=torch.uint8)
    old_output = processor.process_videos(
        [first],
        processor._processor,
        video_metadata=[{"fps": 1, "total_num_frames": 2, "frames_indices": [0, 1]}],
        return_tensors="pt",
    )
    grouped = processor.process_videos(
        [first, second],
        processor._processor,
        return_tensors="pt",
        max_soft_tokens=280,
        source_configs=[
            {"max_soft_tokens": 70},
            {
                "video_metadata": [
                    {"fps": 2, "total_num_frames": 8, "frames_indices": [6]}
                ]
            },
        ],
        video_metadata=[
            {"fps": 1, "total_num_frames": 2, "frames_indices": [0, 1]},
            {"fps": 1, "total_num_frames": 1, "frames_indices": [0]},
        ],
    )
    pixels = grouped.encoder_inputs["pixel_values_videos"]
    positions = grouped.encoder_inputs["video_position_ids"]
    assert isinstance(pixels, torch.Tensor) and isinstance(positions, torch.Tensor)
    assert pixels.shape[:2] == (1, 3)
    assert positions.ne(-1).any(dim=-1).any(dim=-1).all()
    original_patches = old_output.encoder_inputs["pixel_values_videos"].shape[-2]
    torch.testing.assert_close(
        pixels[:, :2, :original_patches],
        old_output.encoder_inputs["pixel_values_videos"],
    )
    assert not pixels[:, :2, original_patches:].any()
    assert positions[:, :2, original_patches:].eq(-1).all()
    assert [
        item.encoder_inputs["pixel_values_videos"].shape[1] for item in grouped.items
    ] == [2, 1]
    assert all(
        item.encoder_inputs["pixel_values_videos"].untyped_storage().data_ptr()
        == pixels.untyped_storage().data_ptr()
        for item in grouped.items
    )
    assert grouped.items[0].metadata == old_output.items[0].metadata
    assert grouped.items[1].metadata["timestamps"] == [3]
    torch.testing.assert_close(
        grouped.items[0].encoder_inputs["pixel_values_videos"],
        old_output.items[0].encoder_inputs["pixel_values_videos"],
    )
    torch.testing.assert_close(
        grouped.items[0].encoder_inputs["video_position_ids"],
        old_output.items[0].encoder_inputs["video_position_ids"],
    )
    direct = processor.process_videos(
        [list(first), list(second)], processor._processor, return_tensors="pt"
    )
    assert direct.encoder_inputs["pixel_values_videos"].shape[:2] == (1, 3)
    assert [item.media_id for item in direct.items] == [("video", 0), ("video", 1)]


@pytest.mark.parametrize("unified", [False, True], ids=["gemma4", "unified"])
def test_audio_groups_pad_native_tensors_and_preserve_source_views(unified):
    processor = _make_processor(unified)
    first = np.zeros(641, dtype=np.float32)
    second = np.zeros(3201, dtype=np.float32)
    history = processor.process_audio(
        [first], processor._processor, padding=True, return_tensors="pt"
    )
    output = processor.process_audio(
        [first, second],
        processor._processor,
        padding=False,
        return_tensors="pt",
        source_configs=[{"padding": True}, {}],
    )
    features = output.encoder_inputs["input_features"]
    mask = output.encoder_inputs["input_features_mask"]
    assert isinstance(features, torch.Tensor) and isinstance(mask, torch.Tensor)
    assert features.shape[0] == mask.shape[0] == 2
    old_length = history.encoder_inputs["input_features"].shape[1]
    torch.testing.assert_close(
        features[0, :old_length], history.encoder_inputs["input_features"][0]
    )
    torch.testing.assert_close(
        mask[0, :old_length], history.encoder_inputs["input_features_mask"][0]
    )
    assert not features[0, old_length:].any()
    assert not mask[0, old_length:].any()
    assert output.items[0].metadata == history.items[0].metadata
    torch.testing.assert_close(
        output.items[0].encoder_inputs["input_features"],
        history.items[0].encoder_inputs["input_features"],
    )
    torch.testing.assert_close(
        output.items[0].encoder_inputs["input_features_mask"],
        history.items[0].encoder_inputs["input_features_mask"],
    )
    assert all(
        item.encoder_inputs["input_features"].untyped_storage().data_ptr()
        == features.untyped_storage().data_ptr()
        for item in output.items
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
