"""CPU parity tests for the shared Qwen media pipeline; no model weights.

    load -> images / video decoder or frames / waveform
                            |
                      process_media -> native encoder inputs
                                               |
    original IDs -> mm_token_expansion <--- grid / timestamp / audio-length rules
                         |
             build -> per-source offsets + Qwen MRoPE

Real HF processors provide independent feature/token references for PIL and NumPy
images. Caller IDs are never decoded, media is processed once, and only timestamp
strings are encoded.
Mixed image/video/audio fields stay isolated. Partial expansion takes trailing
media items, preserves noncanonical history and retains each source tensor view.
Reader tests check process-stage sampling/resize, sampled indices and closure.
Frozen recipes preserve image resize, audio truncation and video frame layout;
grouped image and padded audio items share their final batched tensor storage.
Audio source tensors retain common padded widths for serving concatenation;
zero masks exclude added padding before audio encoding.
HF VideoMetadata remains processor metadata and never enters serving media fields.
Repeated expansion leaves shared frame metadata unchanged.
Non-Qwen subclasses retain their original dispatch and token protocol.
"""

import re
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
    Qwen3OmniMoeProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    WhisperFeatureExtractor,
)

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE, VIDEO, AUDIO, START, END, EOS = 101, 102, 103, 104, 105, 106
TOKEN_TEXT = {
    IMAGE: "<|image_pad|>",
    VIDEO: "<|video_pad|>",
    AUDIO: "<|audio_pad|>",
    START: "<|vision_start|>",
    END: "<|vision_end|>",
    EOS: "<|im_end|>",
}


class _VideoDecoder(VideoDecoderWrapper):
    avg_fps = 2.0

    def __init__(self, frames, fps=2.0):
        self.frames = frames
        self.avg_fps = fps
        self.close_count = 0

    def __len__(self):
        return len(self.frames)

    def get_frames_as_tensor(self, indices):
        return self.frames[indices]

    def close(self):
        self.close_count += 1


@contextmanager
def _ban_text_processing(processor):
    encode = processor._tokenizer.encode

    def encode_timestamp(text, **kwargs):
        if not re.fullmatch(r"<\d+\.\d seconds>", text):
            raise AssertionError(f"Only new timestamp text may be encoded: {text}")
        return encode(text, **kwargs)

    with (
        patch.object(
            type(processor._processor),
            "__call__",
            side_effect=AssertionError("No full HF processor"),
        ),
        patch.object(
            processor._tokenizer,
            "decode",
            side_effect=AssertionError("No prompt decode"),
        ),
        patch.object(processor._tokenizer, "encode", side_effect=encode_timestamp),
    ):
        yield


def _make_processor(model_type="qwen3_vl"):
    processor = object.__new__(QwenVLImageProcessor)
    processor.model_type = model_type
    processor.supports_token_expansion = (
        model_type in QwenVLImageProcessor._TOKENIZED_MEDIA_MODEL_TYPES
    )
    processor.prefer_tokenized_input = processor.supports_token_expansion
    processor._tokenizer = SimpleNamespace(
        eos_token="<|im_end|>",
        eos_token_id=EOS,
        convert_ids_to_tokens=lambda ids: [TOKEN_TEXT[i] for i in ids],
        decode=Mock(side_effect=AssertionError("Caller IDs must not be decoded")),
    )
    processor._processor = SimpleNamespace(tokenizer=processor._tokenizer)
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token_id=IMAGE, video_token_id=VIDEO, audio_token_id=AUDIO
    )
    processor.use_cuda_ipc = False
    processor.precompute_hash_before_cpu_transfer = False
    processor.mm_feature_transport = "cpu"
    processor.FEATURE_NAMES = ["pixel_values", "pixel_values_videos", "input_features"]
    processor.ATTR_NAME_TO_MODALITY = {
        "pixel_values": Modality.IMAGE,
        "image_grid_thw": Modality.IMAGE,
        "pixel_values_videos": Modality.VIDEO,
        "video_grid_thw": Modality.VIDEO,
        "input_features": Modality.AUDIO,
        "feature_attention_mask": Modality.AUDIO,
    }
    processor.hf_config = SimpleNamespace(model_type=processor.model_type)
    processor._spatial_merge_size = 2
    processor._tokens_per_second = 25
    processor.vision_start_token_id = START
    processor.vision_end_token_id = END
    processor.audio_start_token_id = None
    processor.mm_processor_executor = None
    processor.image_config = {}
    processor.video_config = {}
    processor.audio_config = {}
    processor.disable_fast_image_processor = True
    processor._tokenizer_auto_adds_specials = False
    return processor


def _make_hf_processor(processor_class, patch_size, *, model_type="qwen3_vl"):
    # This tokenizer has a real noncanonical sequence: [D, escribe] decodes to
    # Describe, whose canonical encoding is one token.
    tokens = ["[UNK]", "[PAD]", "Describe", "D", "escribe"] + [
        f"unused_{i}" for i in range(5, EOS + 1)
    ]
    for token_id, text in TOKEN_TEXT.items():
        tokens[token_id] = text
    tokens.extend([f"<{tenth / 10:.1f}" for tenth in range(100)] + ["seconds>"])
    vocab = {text: token_id for token_id, text in enumerate(tokens)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=[
            text for token_id, text in TOKEN_TEXT.items() if token_id != EOS
        ],
    )
    processor = _make_processor(model_type)
    processor._tokenizer = tokenizer
    processor_kwargs = {}
    if processor_class is Qwen3OmniMoeProcessor:
        for attr, value in {
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
            "audio_token": "<|audio_pad|>",
            "vision_bos_token": "<|vision_start|>",
            "vision_eos_token": "<|vision_end|>",
            "audio_bos_token": "<|vision_start|>",
            "audio_eos_token": "<|vision_end|>",
        }.items():
            setattr(tokenizer, attr, value)
        processor_kwargs["feature_extractor"] = WhisperFeatureExtractor(
            feature_size=128
        )
    vision_kwargs = {
        "patch_size": patch_size,
        "size": {
            "shortest_edge": (4 * patch_size) ** 2,
            "longest_edge": (8 * patch_size) ** 2,
        },
    }
    processor._processor = processor_class(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**vision_kwargs),
        video_processor=(
            Qwen3VLVideoProcessor
            if processor_class is Qwen3VLProcessor
            else Qwen2VLVideoProcessor
        )(**vision_kwargs),
        **processor_kwargs,
    )
    return processor


class TestQwenTokenizedMedia(CustomTestCase):
    def test_non_qwen_subclass_keeps_legacy_token_processing(self):
        class OtherVLMProcessor(QwenVLImageProcessor):
            pass

        hf_processor = _make_processor()._processor

        def init_base(instance, hf_config, *args, **kwargs):
            instance.__dict__.update(_make_processor(hf_config.model_type).__dict__)
            instance.hf_config = hf_config

        for model_type in (
            "paddleocr_vl",
            "interns1_1",
            "pointsv1.5_chat",
            "intern_s2_preview",
        ):
            with (
                self.subTest(model_type=model_type),
                patch.object(BaseMultimodalProcessor, "__init__", init_base),
            ):
                config = SimpleNamespace(
                    model_type=model_type,
                    vision_start_token_id=START,
                    vision_end_token_id=END,
                    image_token_id=IMAGE,
                    video_token_id=VIDEO,
                    vision_config=SimpleNamespace(spatial_merge_size=2),
                )
                processor = OtherVLMProcessor(config, None, hf_processor)
                self.assertFalse(processor.prefer_tokenized_input)
                original = [3, 4, IMAGE]
                base = BaseMultiModalProcessorOutput(
                    input_text="legacy prompt", input_ids=original, images=[object()]
                )
                processor.get_mm_token_replacements = Mock(
                    side_effect=AssertionError("Qwen token expansion must not run")
                )
                processor._processor = Mock(
                    tokenizer=processor._tokenizer,
                    return_value={"input_ids": torch.tensor([[2, IMAGE]])},
                )
                with patch.dict("os.environ", {"SGLANG_MM_AVOID_RETOKENIZE": "0"}):
                    _, ids, _ = processor.process_and_combine_mm_data(
                        base, processor.mm_tokens
                    )
                self.assertEqual(
                    processor._processor.call_args.kwargs["text"], ["legacy prompt"]
                )
                processor.get_mm_token_replacements.assert_not_called()
                self.assertEqual(ids.tolist(), [2, IMAGE])

    def test_real_images_partial_expansion_and_source_views(self):
        for model_type, hf_class, patch_size in (
            ("qwen2_vl", Qwen2VLProcessor, 14),
            ("qwen2_5_vl", Qwen2_5_VLProcessor, 14),
            ("qwen3_vl", Qwen3VLProcessor, 16),
            ("qwen3_5", Qwen3VLProcessor, 16),
            ("qwen3_5_moe", Qwen3VLProcessor, 16),
        ):
            with self.subTest(model_type=model_type):
                processor = _make_hf_processor(
                    hf_class, patch_size, model_type=model_type
                )
                images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 128))]
                hf = processor._processor
                reference = hf(
                    text="<|vision_start|><|image_pad|><|vision_end|>" * 2,
                    images=images,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                original = [3, 4, EOS, EOS, START, IMAGE, END, START, IMAGE, END]
                with _ban_text_processing(processor):
                    media = processor.process_media(
                        images=[images[0], np.asarray(images[1])]
                    )
                    expanded = processor.mm_token_expansion(original, media)
                    actual = processor.build_multimodal_inputs(
                        expanded, media, consumer="training"
                    )
                    serving = processor.build_multimodal_inputs(expanded, media)
                    first_end = expanded.new_media_bindings[("image", 0)][0][1] + 1
                    history = expanded.input_ids[:first_end]
                    partial = processor.mm_token_expansion(
                        history + [START, IMAGE, END], media, len(history)
                    )
                self.assertEqual(
                    actual["input_ids"].tolist(),
                    [original[:4] + reference["input_ids"][0].tolist()],
                )
                self.assertEqual(partial.input_ids, expanded.input_ids)
                self.assertEqual(set(partial.new_media_bindings), {("image", 1)})
                torch.testing.assert_close(
                    actual["pixel_values"], reference["pixel_values"]
                )
                self.assertEqual(
                    serving.mrope_positions.shape, (3, len(expanded.input_ids))
                )
                for item in media["image"].items:
                    self.assertEqual(
                        item.encoder_inputs["pixel_values"]
                        .untyped_storage()
                        .data_ptr(),
                        media["image"]
                        .encoder_inputs["pixel_values"]
                        .untyped_storage()
                        .data_ptr(),
                    )

    def test_real_video_features_timestamps_and_sampling_intervals(self):
        for model_type, hf_class in (
            ("qwen2_vl", Qwen2VLProcessor),
            ("qwen2_5_vl", Qwen2_5_VLProcessor),
            ("qwen3_vl", Qwen3VLProcessor),
            ("qwen3_5", Qwen3VLProcessor),
            ("qwen3_omni_moe", Qwen3OmniMoeProcessor),
        ):
            with self.subTest(model_type=model_type):
                processor = _make_hf_processor(
                    hf_class,
                    16 if hf_class is Qwen3VLProcessor else 14,
                    model_type=model_type,
                )
                processor.video_config = {"do_sample_frames": False}
                videos = [
                    torch.zeros(4, 3, 64, 64, dtype=torch.uint8),
                    torch.ones(2, 3, 64, 128, dtype=torch.uint8),
                ]
                metadata = [
                    {"fps": 8.0, "total_num_frames": 8, "frames_indices": [0, 2, 4, 6]},
                    {"fps": 6.0, "total_num_frames": 6, "frames_indices": [0, 3]},
                ]
                kwargs = {
                    "video_metadata": metadata,
                    "do_sample_frames": False,
                    "return_metadata": True,
                    "fps": 2.0,
                }
                reference = processor._processor(
                    text="<|video_pad|><|im_end|><|video_pad|>",
                    videos=videos,
                    add_special_tokens=False,
                    return_tensors="pt",
                    **kwargs,
                )
                with _ban_text_processing(processor):
                    media = processor.process_media(videos=videos, **kwargs)
                    expanded = processor.mm_token_expansion([VIDEO, EOS, VIDEO], media)
                    actual = processor.build_multimodal_inputs(
                        expanded, media, consumer="training", return_metadata=True
                    )
                for key in ("input_ids", "pixel_values_videos", "video_grid_thw"):
                    torch.testing.assert_close(actual[key], reference[key])
                for key in ("second_per_grid_ts", "video_second_per_grid"):
                    if key in reference:
                        torch.testing.assert_close(
                            torch.as_tensor(actual[key]), reference[key]
                        )
                if hf_class is Qwen3VLProcessor:
                    self.assertEqual(len(expanded.new_media_bindings[("video", 0)]), 2)
                processor.position_encoding = None
                serving = processor.build_multimodal_inputs(expanded, media)
                for item in serving.mm_items:
                    self.assertNotIn("video_metadata", item.model_specific_data)
                    self.assertIsInstance(item.video_grid_thw, torch.Tensor)

    def test_omni_audio_mask_and_cnn_chunk_lengths(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        processor.audio_config = {"return_attention_mask": True, "truncation": False}
        audios = [np.zeros(3200, dtype=np.float32), np.zeros(19200, dtype=np.float32)]
        reference = processor._processor(
            text="<|audio_pad|><|im_end|><|audio_pad|>",
            audio=audios,
            add_special_tokens=False,
            return_tensors="pt",
            **processor.audio_config,
        )
        with _ban_text_processing(processor):
            media = processor.process_media(audios=audios)
            expanded = processor.mm_token_expansion([AUDIO, EOS, AUDIO], media)
            result = processor.build_multimodal_inputs(
                expanded, media, consumer="training"
            )
        for key in ("input_ids", "input_features", "feature_attention_mask"):
            torch.testing.assert_close(result[key], reference[key])
        self.assertNotIn("attention_mask", media["audio"].encoder_inputs)
        self.assertEqual(
            result["input_ids"].tolist(), [[AUDIO] * 3 + [EOS] + [AUDIO] * 16]
        )

    @patch("sglang.srt.multimodal.processors.qwen_vl.is_cpu", return_value=True)
    def test_video_reader_is_decoded_resized_and_closed_in_process(self, _is_cpu):
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        processor.video_config = {
            "nframes": 2,
            "resized_height": 56,
            "resized_width": 56,
        }
        decoder = _VideoDecoder(torch.zeros(8, 112, 112, 3, dtype=torch.uint8), fps=8)
        output = processor.process_videos(
            [decoder], processor._processor, return_tensors="pt"
        )
        self.assertEqual(decoder.close_count, 1)
        self.assertEqual(
            output.items[0].metadata["video_metadata"].frames_indices.tolist(), [0, 7]
        )
        self.assertEqual(output.encoder_inputs["video_grid_thw"].tolist(), [[1, 4, 4]])

    def test_omni_audio_in_video_is_not_implicitly_enabled(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        with self.assertRaisesRegex(ValueError, "use_audio_in_video"):
            processor.process_media(videos=[object()], use_audio_in_video=True)

    def test_image_recipe_keeps_historical_resize_when_request_defaults_change(self):
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        image = Image.new("RGB", (64, 64))
        first = processor.process_media(images=[image])["image"]
        second = processor.process_media(
            images=[
                {"url": image, "process_options": first.items[0].effective_options},
                image,
            ],
            size={"shortest_edge": 128**2, "longest_edge": 128**2},
        )["image"]
        torch.testing.assert_close(
            first.items[0].encoder_inputs["pixel_values"],
            second.items[0].encoder_inputs["pixel_values"],
        )
        self.assertEqual(second.items[1].metadata["image_grid_thw"].tolist(), [1, 8, 8])
        for item in second.items:
            self.assertEqual(
                item.encoder_inputs["pixel_values"].untyped_storage().data_ptr(),
                second.encoder_inputs["pixel_values"].untyped_storage().data_ptr(),
            )

    def test_video_recipe_and_expansion_do_not_mutate_source_metadata(self):
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        video = torch.zeros(3, 3, 64, 64, dtype=torch.uint8)
        metadata = [{"fps": 3, "total_num_frames": 3, "frames_indices": [0, 1, 2]}]
        first = processor.process_media(
            videos=[video], video_metadata=metadata, do_sample_frames=False
        )
        item = first["video"].items[0]
        indices = list(item.metadata["video_metadata"].frames_indices)
        first_expansion = processor.mm_token_expansion([VIDEO], first)
        second_expansion = processor.mm_token_expansion([VIDEO], first)
        self.assertEqual(first_expansion, second_expansion)
        self.assertEqual(list(item.metadata["video_metadata"].frames_indices), indices)
        replayed = processor.process_media(
            videos=[{"url": video, "process_options": item.effective_options}],
            fps=10,
            size={"shortest_edge": 128**2, "longest_edge": 128**2},
        )
        torch.testing.assert_close(
            first["video"].encoder_inputs["pixel_values_videos"],
            replayed["video"].encoder_inputs["pixel_values_videos"],
        )
        self.assertEqual(
            processor.mm_token_expansion([VIDEO], replayed), first_expansion
        )

    def test_audio_recipe_keeps_truncation_and_mixes_batch_padding(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        audio = np.zeros(3200, dtype=np.float32)
        first = processor.process_media(
            audios=[audio], truncation=True, max_length=1600
        )["audio"]
        second = processor.process_media(
            audios=[
                {"url": audio, "process_options": first.items[0].effective_options},
                audio,
            ],
            truncation=False,
        )["audio"]
        torch.testing.assert_close(
            first.items[0].encoder_inputs["input_features"],
            second.items[0].encoder_inputs["input_features"][
                ..., : first.encoder_inputs["input_features"].shape[-1]
            ],
        )
        self.assertEqual(first.items[0].metadata, second.items[0].metadata)
        self.assertLess(
            second.items[0].metadata["token_count"],
            second.items[1].metadata["token_count"],
        )
        self.assertIsInstance(second.encoder_inputs["input_features"], torch.Tensor)
        original_length = first.encoder_inputs["feature_attention_mask"].shape[-1]
        self.assertEqual(
            second.items[0]
            .encoder_inputs["feature_attention_mask"][..., original_length:]
            .count_nonzero()
            .item(),
            0,
        )
        for name in ("input_features", "feature_attention_mask"):
            torch.testing.assert_close(
                torch.cat([item.encoder_inputs[name] for item in second.items]),
                second.encoder_inputs[name],
            )
        for item in second.items:
            for name in ("input_features", "feature_attention_mask"):
                self.assertEqual(
                    item.encoder_inputs[name].untyped_storage().data_ptr(),
                    second.encoder_inputs[name].untyped_storage().data_ptr(),
                )


if __name__ == "__main__":
    unittest.main()
