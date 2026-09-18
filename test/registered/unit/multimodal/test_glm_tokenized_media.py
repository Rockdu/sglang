"""GLM media boundary tests using local HF processors, without model weights.

    load -> images / video decoder or frames
                         |
                   process_media -> tensor views + grids + timestamps
                                               |
    original IDs -> mm_token_expansion <--- per-source token fragments
                         |
                     shared build -> separate image / nested-video bindings

HF outputs provide independent numeric/token references. Partial expansion reads
only the new suffix; build independently rejects changed historical layouts.

    first video -> process -> frozen recipe
    old(recipe) + new -> retain old budget/frames -> budget only the new video

Actual grids validate budgets. Reader tests check process-stage sampling and closure.
Video processing preserves source pixels and resolves missing FPS before timestamp
expansion. No text decoding occurs on the ID route.
Historical layout fixtures retain the component's patch tensor contract.
"""

import asyncio
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.media_processing import (
    collect_media_bindings,
    pack_grid_media_output,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.glm4v import (
    Glm4vImageProcessor,
    _process_glm_video,
)
from sglang.srt.multimodal.processors.glm_image import GlmImageProcessor
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

IMAGE, VIDEO, IMAGE_START, IMAGE_END, VIDEO_START, VIDEO_END = range(101, 107)


class _NativeFramesDecoder(VideoDecoderWrapper):
    avg_fps = 4

    def __init__(self, frames):
        self.frames = frames
        self.decoded_frame_indices = []
        self.closed = False

    def __len__(self):
        return len(self.frames)

    def get_frames_as_tensor(self, indices):
        assert not self.closed
        self.decoded_frame_indices.append(indices)
        return self.frames[indices]

    def close(self):
        self.closed = True


def _base_settings(processor):
    from transformers.models.glm4v.processing_glm4v import (
        Glm4vProcessor,
        Glm4vProcessorKwargs,
    )

    processor._tokenizer = SimpleNamespace(
        init_kwargs={},
        decode=Mock(side_effect=AssertionError("Caller tokens must never be decoded")),
        encode=Mock(side_effect=lambda text, **kwargs: [40 + int(text)]),
    )
    processor._processor = Mock(
        spec=[
            "tokenizer",
            "image_processor",
            "video_processor",
            "valid_processor_kwargs",
            "_merge_kwargs",
            "unused_input_names",
            "skip_tensor_conversion",
        ],
        tokenizer=processor._tokenizer,
        image_processor=Mock(spec=["merge_size"], merge_size=1),
        video_processor=Mock(spec=["merge_size"], merge_size=1),
        valid_processor_kwargs=Glm4vProcessorKwargs,
        unused_input_names=[],
        skip_tensor_conversion=["video_metadata", "text_replacement_offsets"],
    )
    processor._processor._merge_kwargs = partial(
        Glm4vProcessor._merge_kwargs, processor._processor
    )
    processor._tokenizer_auto_adds_specials = False
    processor.image_config = {}
    processor.video_config = {}
    processor.audio_config = {}
    processor.disable_fast_image_processor = True
    processor.use_cuda_ipc = False
    processor.precompute_hash_before_cpu_transfer = False
    processor.mm_feature_transport = "cpu"
    processor.ATTR_NAME_TO_MODALITY = {
        "pixel_values": Modality.IMAGE,
        "image_grid_thw": Modality.IMAGE,
        "pixel_values_videos": Modality.VIDEO,
        "video_grid_thw": Modality.VIDEO,
    }
    processor.FEATURE_NAMES = ["pixel_values", "pixel_values_videos"]
    return processor


def _vision_processor():
    processor = _base_settings(object.__new__(Glm4vImageProcessor))
    processor.IM_TOKEN_ID, processor.VIDEO_TOKEN_ID = IMAGE, VIDEO
    processor.IMAGE_START_TOKEN_ID, processor.IMAGE_END_TOKEN_ID = (
        IMAGE_START,
        IMAGE_END,
    )
    processor.VIDEO_START_TOKEN_ID, processor.VIDEO_END_TOKEN_ID = (
        VIDEO_START,
        VIDEO_END,
    )
    processor.IMAGE_TOKEN, processor.VIDEO_TOKEN = "<|image|>", "<|video|>"
    processor.IMAGE_START_TOKEN, processor.IMAGE_END_TOKEN = (
        "<|begin_of_image|>",
        "<|end_of_image|>",
    )
    processor.VIDEO_START_TOKEN, processor.VIDEO_END_TOKEN = (
        "<|begin_of_video|>",
        "<|end_of_video|>",
    )
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token_id=IMAGE, video_token_id=IMAGE
    )
    return processor


def _local_hf_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"token_{i}": i for i in range(107)}
    special_tokens = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|image|>": IMAGE,
        "<|video|>": VIDEO,
        "<|begin_of_image|>": IMAGE_START,
        "<|end_of_image|>": IMAGE_END,
        "<|begin_of_video|>": VIDEO_START,
        "<|end_of_video|>": VIDEO_END,
    }
    for token_id, token in enumerate([*map(str, range(10)), ".", "seconds"], 60):
        del vocab[f"token_{token_id}"]
        vocab[token] = token_id
    for token, token_id in special_tokens.items():
        del vocab[f"token_{token_id}"]
        vocab[token] = token_id
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=list(special_tokens)[2:],
    )


def _attach_real_hf_processor(processor, hf):
    processor._processor = hf
    processor._tokenizer = hf.tokenizer
    return processor


@pytest.mark.parametrize("family", ["glm4v", "glm46v"])
@pytest.mark.parametrize("fps", [3, None], ids=["known-fps", "missing-fps"])
def test_real_hf_glm_image_and_video_preprocessing_preserves_arbitrary_tokens(
    family, fps, monkeypatch
):
    from PIL import Image

    if family == "glm4v":
        from transformers.models.glm4v.image_processing_glm4v import (
            Glm4vImageProcessor as HFImageProcessor,
        )
        from transformers.models.glm4v.processing_glm4v import (
            Glm4vProcessor as HFProcessor,
        )
        from transformers.models.glm4v.video_processing_glm4v import (
            Glm4vVideoProcessor as HFVideoProcessor,
        )
    else:
        from transformers.models.glm46v.image_processing_glm46v import (
            Glm46VImageProcessor as HFImageProcessor,
        )
        from transformers.models.glm46v.processing_glm46v import (
            Glm46VProcessor as HFProcessor,
        )
        from transformers.models.glm46v.video_processing_glm46v import (
            Glm46VVideoProcessor as HFVideoProcessor,
        )

    hf = HFProcessor(
        tokenizer=_local_hf_tokenizer(),
        image_processor=HFImageProcessor(),
        video_processor=HFVideoProcessor(),
    )
    processor = _attach_real_hf_processor(_vision_processor(), hf)
    original = [7, 8, 8, VIDEO_START, VIDEO, VIDEO_END] + [
        9,
        IMAGE_START,
        IMAGE,
        IMAGE_END,
        10,
    ]
    base = BaseMultiModalProcessorOutput(
        input_text="",
        input_ids=original,
        images=[Image.new("RGB", (112, 112))],
        videos=[torch.zeros(4, 3, 112, 112, dtype=torch.uint8)],
    )
    metadata = [
        {
            "total_num_frames": 8,
            "fps": fps,
            "duration": 8 / (fps or 24),
            "frames_indices": [1, 2, 4, 7],
        }
    ]
    reference = hf(
        text=[
            "<|begin_of_video|><|video|><|end_of_video|>token_9"
            "<|begin_of_image|><|image|><|end_of_image|>"
        ],
        images=base.images,
        videos=base.videos,
        video_metadata=metadata,
        return_metadata=True,
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
        device="cpu",
    )
    for owner, method in (
        (type(hf), "__call__"),
        (type(hf), "replace_image_token"),
        (type(hf), "replace_video_token"),
        (type(hf.tokenizer), "__call__"),
        (hf.tokenizer, "decode"),
    ):
        monkeypatch.setattr(
            owner,
            method,
            Mock(side_effect=AssertionError("Token path used text route")),
        )
    timestamp_encode = Mock(wraps=hf.tokenizer.encode)
    monkeypatch.setattr(hf.tokenizer, "encode", timestamp_encode)
    media = processor.process_media(
        images=base.images,
        videos=base.videos,
        video_metadata=metadata,
        return_metadata=True,
        do_sample_frames=False,
    )
    expanded = processor.mm_token_expansion(original, media)
    processor_output = processor.build_multimodal_inputs(
        expanded, media, consumer="training", return_metadata=True
    )
    bindings = collect_media_bindings(
        expanded.input_ids,
        processor.get_mm_token_replacements(hf, media),
        expanded.new_media_bindings,
    )
    expanded_input_ids = expanded.input_ids
    assert expanded_input_ids == [7, 8, 8] + reference["input_ids"][0].tolist() + [10]
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ):
        assert torch.equal(processor_output[key], reference[key])
    expected_timestamp_text = {
        ("glm4v", 3): ["0", "1"],
        ("glm4v", None): ["0", "0"],
        ("glm46v", 3): ["0.3 seconds", "1.3 seconds"],
        ("glm46v", None): ["0.0 seconds", "0.2 seconds"],
    }[(family, fps)]
    assert [
        call.args[0] for call in timestamp_encode.call_args_list[:2]
    ] == expected_timestamp_text
    returned_metadata = processor_output["video_metadata"][0]
    assert returned_metadata.fps == (fps or 24)
    assert returned_metadata.timestamps == reference["video_metadata"][0].timestamps
    assert expanded_input_ids[:4] == original[:4]
    assert expanded_input_ids[-1] == 10
    video_end = expanded_input_ids.index(VIDEO_END)
    assert expanded_input_ids[video_end + 1 : video_end + 3] == [9, IMAGE_START]
    assert expanded_input_ids[video_end + 3 : -2] == [IMAGE] * 16
    assert media["image"].items[0].encoder_inputs["pixel_values"].shape == (64, 1176)
    assert media["video"].items[0].encoder_inputs["pixel_values_videos"].shape[0] == 128
    assert len(bindings[("image", 0)]) == 1
    assert len(bindings[("video", 0)]) == 2
    assert all(
        expanded_input_ids[start:end] == [IMAGE] * (end - start)
        for spans in bindings.values()
        for start, end in spans
    )
    history = expanded_input_ids.copy()
    video_recipe = media["video"].items[0].effective_options
    media = processor.process_media(
        images=base.images + [Image.new("RGB", (112, 112), "white")],
        videos=[{"url": base.videos[0], "process_options": video_recipe}],
        fps=10,
        do_sample_frames=False,
    )
    partial = processor.mm_token_expansion(
        history + [7, IMAGE_START, IMAGE, IMAGE_END], media, len(history)
    )
    assert partial.input_ids == history + [7, IMAGE_START] + [IMAGE] * 16 + [IMAGE_END]
    assert set(partial.new_media_bindings) == {("image", 1)}
    processor.build_multimodal_inputs(partial, media, consumer="training")


def test_glm_supplied_video_frame_dictionaries_preserve_pixels_and_timestamps():
    images = torch.zeros(4, 3, 56, 84, dtype=torch.uint8)
    frames = [
        {
            "frame_image": image,
            "timestamp": index / 2,
            "detail": '{"video_duration": 2}',
        }
        for index, image in enumerate(images)
    ]
    loaded, metadata = _process_glm_video(frames, {}, video_processor=None)
    assert torch.equal(loaded, images.permute(0, 2, 3, 1))
    assert metadata["frames_indices"] == [0, 1, 2, 3]
    assert metadata["fps"] == 2
    assert metadata["duration"] == 2


def test_glm_image_preprocessed_source_and_target_grids_stay_separate():
    processor = _base_settings(object.__new__(GlmImageProcessor))
    (
        processor.IM_TOKEN_ID,
        processor.IMAGE_START_TOKEN_ID,
        processor.IMAGE_END_TOKEN_ID,
    ) = IMAGE, IMAGE_START, IMAGE_END
    processor.mm_tokens = MultimodalSpecialTokens(image_token_id=IMAGE)
    original = [7, IMAGE_START, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE_END, 8]
    full_grid = torch.tensor([[1, 2, 2], [1, 3, 3]])
    source_pixels = torch.zeros(4, 8)
    output = asyncio.run(
        processor.process_mm_data_async(
            [{"pixel_values": source_pixels, "image_grid_thw": full_grid}],
            original,
            SimpleNamespace(),
        )
    )
    assert output.input_ids == original
    assert output.mm_items[0].image_grid_thw.tolist() == [[1, 2, 2]]
    assert output.mm_items[0].offsets == [(2, 5)]
    assert output.mm_items[0].feature is source_pixels
    # The target contributes decode positions, never source encoder patches.
    assert output.mrope_positions.shape == (3, len(original) + 3 * 3 + 1)
    assert full_grid.tolist() == [[1, 2, 2], [1, 3, 3]]


class _BudgetVideoProcessor:
    merge_size = 1
    max_image_tokens = 32
    min_image_tokens = 0
    patch_size = 14
    temporal_patch_size = 2

    def __call__(self, videos, *, video_metadata, **kwargs):
        count = kwargs.get("max_image_tokens", self.max_image_tokens)
        return {
            "pixel_values_videos": torch.ones(len(videos) * count, 1),
            "video_grid_thw": torch.tensor([[1, 1, count] for _ in videos]),
            "video_metadata": [
                SimpleNamespace(**metadata) for metadata in video_metadata
            ],
        }


def test_frozen_video_recipe_preserves_history_and_only_new_video_gets_budget():
    processor = _vision_processor()
    processor._processor.video_processor = _BudgetVideoProcessor()
    video = torch.zeros(2, 16, 16, 3, dtype=torch.uint8)
    first = processor.process_videos([video], processor._processor, max_image_tokens=16)
    recipe = first.items[0].effective_options
    second = processor.process_videos(
        [video, video],
        processor._processor,
        process_options=[recipe, None],
        max_image_tokens=24,
    )
    assert (
        first.items[0].metadata["video_grid_thw"].tolist()
        == second.items[0].metadata["video_grid_thw"].tolist()
    )
    torch.testing.assert_close(
        first.items[0].encoder_inputs["pixel_values_videos"],
        second.items[0].encoder_inputs["pixel_values_videos"],
    )
    assert [item.effective_options["token_count"] for item in second.items] == [16, 8]
    import json

    json.dumps([item.effective_options for item in second.items])
    with pytest.raises(ValueError, match="No token budget"):
        processor.process_videos(
            [video, video],
            processor._processor,
            process_options=[recipe, None],
            max_image_tokens=16,
        )


def test_actual_grid_budget_is_checked_after_processing(monkeypatch):
    processor = _vision_processor()
    video_processor = _BudgetVideoProcessor()
    call = _BudgetVideoProcessor.__call__

    def exceed(self, videos, **kwargs):
        kwargs["max_image_tokens"] += 1
        return call(self, videos, **kwargs)

    monkeypatch.setattr(_BudgetVideoProcessor, "__call__", exceed)
    processor._processor.video_processor = video_processor
    with pytest.raises(ValueError, match="exceeds its available token budget"):
        processor.process_videos(
            [torch.zeros(2, 16, 16, 3)], processor._processor, max_image_tokens=8
        )


def test_changed_historical_video_is_checked_by_build_not_expansion():
    processor = _vision_processor()

    def media(grid):
        return {
            "video": pack_grid_media_output(
                "video",
                {
                    "pixel_values_videos": torch.zeros(grid[0] * grid[1] * grid[2], 1),
                    "video_grid_thw": torch.tensor([grid]),
                    "video_metadata": [SimpleNamespace(fps=2, frames_indices=[0, 1])],
                },
                "pixel_values_videos",
                "video_grid_thw",
                metadata_names=("video_metadata",),
            )
        }

    first = processor.mm_token_expansion(
        [VIDEO_START, VIDEO, VIDEO_END], media([1, 1, 2])
    )
    changed = media([1, 1, 3])
    unchanged = processor.mm_token_expansion(
        first.input_ids, changed, len(first.input_ids)
    )
    assert unchanged.input_ids == first.input_ids
    assert unchanged.new_media_bindings == {}
    with pytest.raises(ValueError, match="Historical|historical"):
        processor.build_multimodal_inputs(unchanged, changed, consumer="training")


def test_reader_sampling_is_in_process_and_closes_reader():
    processor = _vision_processor()
    processor._processor.video_processor = _BudgetVideoProcessor()
    processor.video_config = {"fps": 2, "max_frames": 2}
    decoder = _NativeFramesDecoder(torch.zeros(8, 16, 16, 3, dtype=torch.uint8))
    output = processor.process_videos(
        [decoder], processor._processor, max_image_tokens=8
    )
    assert decoder.closed
    assert decoder.decoded_frame_indices
    assert (
        output.items[0].effective_options["video_config"]["frame_indices"]
        == decoder.decoded_frame_indices[0]
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
