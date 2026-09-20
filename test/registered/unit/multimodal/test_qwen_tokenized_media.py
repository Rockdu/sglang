"""CPU parity tests for the shared Qwen media pipeline; no model weights.

    load -> images / video decoder or frames / waveform
                            |
                      process_media -> flat native encoder fields
                                               |
    original IDs -> mm_token_expansion <--- pure pattern -> token sequence specs
                         |
             final IDs + native fields -> collect -> offsets -> serving split
                                               |
                               Qwen hooks -> MRoPE + encoder-DP marking
                                               |
                                      hashes -> feature transport

    startup-selected native -> scheduler grid items -> MRoPE
                            -> EPD embeddings + video times -> IDs / spans / MRoPE

Real HF processors provide feature/token references without model weights.
Partial expansion takes trailing media, preserves history and never decodes IDs.
Reader/frame batches retain native sampling, resize, channels and reader closure.
Grouped video and padded audio retain source order, masks and shared tensor views.
Serving keeps legacy video bundling; adjacent image spans split by expansion lengths.
Bare-placeholder fixtures stub position building; complete prompts exercise MRoPE.
Native encoder-DP flags precede transport; the legacy hook leaves items unchanged.
Repeated expansion preserves metadata; serving excludes HF VideoMetadata.
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
from torchvision.transforms.functional import InterpolationMode, resize
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
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessorKwargs,
)

from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING, get_mm_processor
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
    preprocess_video_sync,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
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
        self.decode_requests = []

    def __len__(self):
        return len(self.frames)

    def get_frames_as_tensor(self, indices):
        self.decode_requests.append(list(indices))
        return self.frames[indices]

    @property
    def frame_shape(self):
        return tuple(self.frames.shape[1:3])

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
    processor.use_token_space_processor = processor.supports_token_expansion
    processor.prefer_tokenized_input = processor.use_token_space_processor
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
    processor.IM_TOKEN_ID = processor.image_token_id = IMAGE
    processor.VIDEO_TOKEN_ID = processor.video_token_id = VIDEO
    processor.audio_token_id = AUDIO
    processor.IM_START_TOKEN_ID = START
    processor.IM_END_TOKEN_ID = END
    if model_type == "qwen3_omni_moe":
        processor.media_processor_kwargs_type = Qwen3OmniMoeProcessorKwargs
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
    def test_encoder_dp_marking_precedes_transport_only_on_the_native_route(self):
        reset_context()
        self.addCleanup(reset_context)
        for native_route, transport, encoder_dp, expected_flags in (
            (True, "cuda_ipc", True, [True, True, False]),
            (True, "cuda_vmm", True, [True, True, False]),
            (True, "cpu", True, [False, False, False]),
            (True, "cuda_ipc", False, [False, False, False]),
            (False, "cuda_ipc", True, [False, False, False]),
        ):
            with self.subTest(
                native_route=native_route, transport=transport, encoder_dp=encoder_dp
            ):
                publish(
                    ServerArgs(model_path="dummy", mm_enable_dp_encoder=encoder_dp),
                    role="test",
                )
                processor = _make_processor()
                processor.use_token_space_processor = native_route
                processor.mm_feature_transport = transport
                processor.use_cuda_ipc = transport == "cuda_ipc"
                items = [
                    MultimodalDataItem(modality=modality, feature=torch.ones(1, 2))
                    for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
                ]

                def check_before_transport(mm_items):
                    self.assertEqual(
                        [
                            item.model_specific_data.get(
                                DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY, False
                            )
                            for item in mm_items
                        ],
                        expected_flags,
                    )
                    return mm_items

                with patch.object(
                    processor,
                    "_prepare_mm_items_for_transport",
                    side_effect=check_before_transport,
                ) as prepare_transport:
                    finalized = processor._finalize_mm_items(items, images=None)
                self.assertIs(finalized, items)
                prepare_transport.assert_called_once_with(items)

    def test_selected_native_processor_preserves_scheduler_and_epd_hooks(self):
        legacy = _make_hf_processor(Qwen3VLProcessor, 16)
        config = SimpleNamespace(
            architectures=["Qwen3VLForConditionalGeneration"],
            model_type="qwen3_vl",
            image_token_id=IMAGE,
            video_token_id=VIDEO,
            vision_start_token_id=START,
            vision_end_token_id=END,
            vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=25),
        )
        reset_context()
        self.addCleanup(reset_context)
        server_args = ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            mm_process_config={},
            enable_token_space_processor=True,
            mm_processor_worker_num=1,
        )
        publish(server_args, role="test")
        with patch.dict(
            PROCESSOR_MAPPING,
            {Qwen3VLForConditionalGeneration: QwenVLImageProcessor},
            clear=True,
        ):
            native = get_mm_processor(
                config,
                server_args,
                legacy._processor,
                None,
            )
        self.addCleanup(native.shutdown)
        self.assertIs(type(native), QwenVLImageProcessor.token_space_processor_class)

        scheduler_ids = [START] + [IMAGE] * 4 + [END, START] + [IMAGE] * 8 + [END]
        scheduler_items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                model_specific_data={"image_grid_thw": torch.tensor([grid])},
            )
            for grid in ([1, 4, 4], [1, 4, 8])
        ]
        positions, delta = native.compute_mrope_positions(
            scheduler_ids, scheduler_items
        )
        expected_positions, expected_delta = legacy.compute_mrope_positions(
            scheduler_ids, scheduler_items
        )
        self.assertEqual(positions.shape, (3, len(scheduler_ids)))
        torch.testing.assert_close(positions, expected_positions)
        torch.testing.assert_close(delta, expected_delta)

        prompt = [START, IMAGE, END, START, VIDEO, END]
        embeddings = {
            Modality.IMAGE: torch.arange(16).reshape(4, 4),
            Modality.VIDEO: torch.arange(32).reshape(8, 4),
        }
        media_kwargs = dict(
            img_grid_thw=torch.tensor([[1, 4, 4]]),
            video_grid_thw=torch.tensor([[2, 4, 4]]),
            video_timestamps=[[0.0, 1.0]],
        )
        actual = native.get_validated_mm_data(prompt, embeddings, **media_kwargs)
        expected = legacy.get_validated_mm_data(prompt, embeddings, **media_kwargs)
        self.assertEqual(actual.input_ids, expected.input_ids)
        self.assertEqual(len(actual.mm_items), 3)
        for actual_item, expected_item in zip(actual.mm_items, expected.mm_items):
            self.assertEqual(actual_item.offsets, expected_item.offsets)
            torch.testing.assert_close(
                actual_item.precomputed_embeddings,
                expected_item.precomputed_embeddings,
            )
        torch.testing.assert_close(actual.mrope_positions, expected.mrope_positions)
        torch.testing.assert_close(
            actual.mrope_position_delta, expected.mrope_position_delta
        )

    def assert_tensor_bytes_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            actual.contiguous().view(torch.uint8).numpy().tobytes()
            == expected.contiguous().view(torch.uint8).numpy().tobytes(),
            "Tensor bytes differ",
        )

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
                patch.object(QwenVLImageProcessor, "_initialize_processor", init_base),
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
                processor.get_mm_token_expansion_spec = Mock(
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
                processor.get_mm_token_expansion_spec.assert_not_called()
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
                    expanded = processor.mm_token_expansion(
                        original,
                        processor.get_mm_token_expansion_spec(
                            processor._processor, media
                        ),
                    )
                    serving = processor.sglang_post_process(expanded, media)
                    first_end = serving.mm_items[0].offsets[0][1] + 2
                    history = expanded[:first_end]
                    partial = processor.mm_token_expansion(
                        history + [START, IMAGE, END],
                        processor.get_mm_token_expansion_spec(
                            processor._processor, media
                        ),
                        len(history),
                    )
                    partial_serving = processor.sglang_post_process(partial, media)
                self.assertEqual(
                    [expanded],
                    [original[:4] + reference["input_ids"][0].tolist()],
                )
                self.assertEqual(partial, expanded)
                self.assertEqual(
                    [item.offsets for item in partial_serving.mm_items],
                    [item.offsets for item in serving.mm_items],
                )
                torch.testing.assert_close(
                    media["pixel_values"], reference["pixel_values"]
                )
                self.assertEqual(serving.mrope_positions.shape, (3, len(expanded)))
                with patch.object(processor, "_build_position_inputs", return_value={}):
                    adjacent = processor.sglang_post_process(
                        [IMAGE]
                        * sum(
                            end - start + 1
                            for item in serving.mm_items
                            for start, end in item.offsets
                        ),
                        media,
                    )
                self.assertEqual(len(adjacent.mm_items), 2)
                self.assertEqual(
                    adjacent.mm_items[0].offsets[0][1] + 1,
                    adjacent.mm_items[1].offsets[0][0],
                )
                for item in serving.mm_items:
                    self.assertEqual(
                        item.feature.untyped_storage().data_ptr(),
                        media["pixel_values"].untyped_storage().data_ptr(),
                    )

    def test_real_video_features_timestamps_and_sampling_intervals(self):
        for model_type, hf_class in (
            ("qwen2_vl", Qwen2VLProcessor),
            ("qwen2_5_vl", Qwen2_5_VLProcessor),
            ("qwen3_vl", Qwen3VLProcessor),
            ("qwen3_5", Qwen3VLProcessor),
            ("qwen4_exp", Qwen3VLProcessor),
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
                    torch.arange(4 * 3 * 112 * 168)
                    .remainder(251)
                    .to(torch.uint8)
                    .reshape(4, 3, 112, 168),
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
                    expanded = processor.mm_token_expansion(
                        [VIDEO, EOS, VIDEO],
                        processor.get_mm_token_expansion_spec(
                            processor._processor, media
                        ),
                    )
                self.assertEqual(expanded, reference["input_ids"][0].tolist())
                for key in ("pixel_values_videos", "video_grid_thw"):
                    self.assert_tensor_bytes_equal(media[key], reference[key])
                for key in ("second_per_grid_ts", "video_second_per_grid"):
                    if key in reference:
                        torch.testing.assert_close(
                            torch.as_tensor(media[key]), reference[key]
                        )
                with patch.object(processor, "_build_position_inputs", return_value={}):
                    serving = processor.sglang_post_process(expanded, media)
                if hf_class is Qwen3VLProcessor:
                    self.assertEqual(
                        [len(item.offsets) for item in serving.mm_items], [2, 1]
                    )
                else:
                    self.assertEqual(len(serving.mm_items), 1)
                    self.assertEqual(len(serving.mm_items[0].offsets), 2)
                self.assert_tensor_bytes_equal(
                    torch.cat([item.feature for item in serving.mm_items]),
                    media["pixel_values_videos"],
                )
                self.assert_tensor_bytes_equal(
                    torch.cat([item.video_grid_thw for item in serving.mm_items]),
                    media["video_grid_thw"],
                )
                for item in serving.mm_items:
                    self.assertNotIn("video_metadata", item.model_specific_data)
                    self.assertIsInstance(item.video_grid_thw, torch.Tensor)

    def test_video_option_groups_preserve_source_order_and_serving_views(self):
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        videos = [
            torch.full((frames, 3, 64, 64), value, dtype=torch.uint8)
            for frames, value in ((2, 31), (4, 127), (2, 223))
        ]
        references = [
            processor.process_videos(
                [video],
                processor._processor,
                do_resize=False,
                do_sample_frames=False,
                do_normalize=normalize,
                return_tensors="pt",
            )
            for video, normalize in zip(videos, (True, False, True))
        ]
        grouped = processor.process_videos(
            videos,
            processor._processor,
            source_configs=[{"do_normalize": value} for value in (True, False, True)],
            do_resize=False,
            do_sample_frames=False,
            return_tensors="pt",
        )
        for name in ("pixel_values_videos", "video_grid_thw"):
            self.assert_tensor_bytes_equal(
                grouped[name],
                torch.cat([source[name] for source in references]),
            )
        media = grouped
        expanded = processor.mm_token_expansion(
            [VIDEO, EOS, VIDEO, EOS, VIDEO],
            processor.get_mm_token_expansion_spec(processor._processor, media),
        )
        with patch.object(processor, "_build_position_inputs", return_value={}):
            serving = processor.sglang_post_process(expanded, media)
        self.assertEqual([len(item.offsets) for item in serving.mm_items], [1, 2, 1])
        for item, source in zip(serving.mm_items, references):
            self.assert_tensor_bytes_equal(item.feature, source["pixel_values_videos"])
            self.assert_tensor_bytes_equal(
                item.video_grid_thw, source["video_grid_thw"]
            )
            self.assertEqual(
                item.feature.untyped_storage().data_ptr(),
                grouped["pixel_values_videos"].untyped_storage().data_ptr(),
            )

    def test_omni_audio_mask_and_cnn_chunk_lengths(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        processor.audio_config = {
            "return_attention_mask": True,
            "truncation": False,
        }
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
            expanded = processor.mm_token_expansion(
                [AUDIO, EOS, AUDIO],
                processor.get_mm_token_expansion_spec(processor._processor, media),
            )
        self.assertEqual(expanded, reference["input_ids"][0].tolist())
        for key in ("input_features", "feature_attention_mask"):
            torch.testing.assert_close(media[key], reference[key])
        self.assertNotIn("attention_mask", media)
        self.assertEqual([expanded], [[AUDIO] * 3 + [EOS] + [AUDIO] * 16])

    def test_native_video_reader_matches_raw_frames(self):
        # Both source dimensions align to legacy factor 28, but not native factor 32.
        frames = (
            torch.arange(8 * 112 * 168 * 3)
            .remainder(251)
            .to(torch.uint8)
            .reshape(8, 112, 168, 3)
        )
        metadata = [{"fps": 8, "total_num_frames": 8, "frames_indices": [0, 2, 4, 7]}]
        for model_type in ("qwen3_5", "qwen4_exp"):
            with self.subTest(model_type=model_type):
                processor = _make_hf_processor(
                    Qwen3VLProcessor, 16, model_type=model_type
                )
                processor.video_config = {"nframes": 4}
                reference = processor._processor(
                    text="<|video_pad|>",
                    videos=[frames[[0, 2, 4, 7]]],
                    video_metadata=metadata,
                    do_sample_frames=False,
                    return_metadata=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                    input_data_format="channels_last",
                )
                decoder = _VideoDecoder(frames, fps=8)
                with (
                    patch(
                        "sglang.srt.multimodal.token_space.qwen_vl.MAX_VIDEO_DECODE_CHUNK_BYTES",
                        2 * 112 * 168 * 3,
                    ),
                    _ban_text_processing(processor),
                ):
                    media = processor.process_media(
                        videos=[decoder], input_data_format="channels_last"
                    )
                    expanded = processor.mm_token_expansion(
                        [VIDEO],
                        processor.get_mm_token_expansion_spec(
                            processor._processor, media
                        ),
                    )
                self.assertEqual(decoder.decode_requests, [[0, 2], [4, 7]])
                self.assertEqual(decoder.close_count, 1)
                self.assertEqual(expanded, reference["input_ids"][0].tolist())
                for key in ("pixel_values_videos", "video_grid_thw"):
                    self.assert_tensor_bytes_equal(media[key], reference[key])
                returned_metadata = media["video_metadata"][0]
                self.assertEqual(list(returned_metadata.frames_indices), [0, 2, 4, 7])
                self.assertEqual(returned_metadata.fps, 8)

    def test_mixed_video_sources_reach_hf_after_one_native_resize(self):
        # This aspect ratio shrinks again if native smart_resize runs twice.
        frames = (
            torch.arange(2 * 32 * 3840 * 3)
            .remainder(251)
            .to(torch.uint8)
            .reshape(2, 32, 3840, 3)
        )
        video_processor = Qwen3VLVideoProcessor(
            size={"shortest_edge": 16384, "longest_edge": 131072}
        )
        decoder = _VideoDecoder(frames)
        prepared, metadata = zip(
            *[
                preprocess_video_sync(
                    source,
                    video_config={"nframes": 2},
                    video_processor=video_processor,
                    processor_kwargs={"input_data_format": "channels_last"},
                    resize_raw_frames=True,
                )
                for source in (decoder, frames)
            ]
        )
        reference = video_processor(
            [frames, frames],
            input_data_format="channels_last",
            do_sample_frames=False,
            return_tensors="pt",
        )
        actual = video_processor(
            list(prepared),
            video_metadata=list(metadata),
            input_data_format="channels_first",
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
        )
        self.assertEqual(actual["video_grid_thw"].tolist(), [[1, 2, 174], [1, 2, 174]])
        for key in ("pixel_values_videos", "video_grid_thw"):
            self.assert_tensor_bytes_equal(actual[key], reference[key])

    @patch("sglang.srt.utils.is_cpu", return_value=True)
    def test_qwen2_video_reader_keeps_legacy_resize(self, _is_cpu):
        processor = _make_hf_processor(Qwen2VLProcessor, 14, model_type="qwen2_vl")
        processor.video_config = {
            "nframes": 2,
            "resized_height": 56,
            "resized_width": 56,
        }
        frames = (
            torch.arange(8 * 112 * 112 * 3)
            .remainder(251)
            .to(torch.uint8)
            .reshape(8, 112, 112, 3)
        )
        decoder = _VideoDecoder(frames, fps=8)
        reference = processor._processor.video_processor(
            [
                resize(
                    frames[[0, 7]].permute(0, 3, 1, 2),
                    [56, 56],
                    interpolation=InterpolationMode.BILINEAR,
                )
            ],
            do_sample_frames=False,
            return_tensors="pt",
        )
        output = processor.process_videos(
            [decoder], processor._processor, return_tensors="pt"
        )
        self.assertEqual(decoder.close_count, 1)
        self.assertEqual(decoder.decode_requests, [[0, 7]])
        self.assertEqual(output["video_metadata"][0].frames_indices.tolist(), [0, 7])
        self.assertEqual(output["video_grid_thw"].tolist(), [[1, 4, 4]])
        for key in ("pixel_values_videos", "video_grid_thw"):
            self.assert_tensor_bytes_equal(output[key], reference[key])

    def test_omni_audio_in_video_is_not_implicitly_enabled(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        with self.assertRaisesRegex(ValueError, "use_audio_in_video"):
            processor.process_media(videos=[object()], use_audio_in_video=True)

    def test_image_source_options_preserve_independent_resize_and_serving_views(self):
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        image = Image.new("RGB", (64, 64))
        image_options = {"size": {"shortest_edge": 64**2, "longest_edge": 64**2}}
        first = processor.process_media(images=[image], **image_options)
        second = processor.process_media(
            images=[
                {"url": image, "preprocess_kwargs": image_options},
                image,
            ],
            size={"shortest_edge": 128**2, "longest_edge": 128**2},
        )
        expanded = processor.mm_token_expansion(
            [START, IMAGE, END] * 2,
            processor.get_mm_token_expansion_spec(processor._processor, second),
        )
        serving = processor.sglang_post_process(expanded, second)
        torch.testing.assert_close(first["pixel_values"], serving.mm_items[0].feature)
        self.assertEqual(second["image_grid_thw"][1].tolist(), [1, 8, 8])
        for item in serving.mm_items:
            self.assertEqual(
                item.feature.untyped_storage().data_ptr(),
                second["pixel_values"].untyped_storage().data_ptr(),
            )

    def test_repeated_video_expansion_does_not_mutate_source_metadata(self):
        processor = _make_hf_processor(Qwen3VLProcessor, 16)
        video = torch.zeros(3, 3, 64, 64, dtype=torch.uint8)
        metadata = [{"fps": 3, "total_num_frames": 3, "frames_indices": [0, 1, 2]}]
        first = processor.process_media(
            videos=[video], video_metadata=metadata, do_sample_frames=False
        )
        metadata = first["video_metadata"][0]
        indices = list(metadata.frames_indices)
        first_expansion = processor.mm_token_expansion(
            [VIDEO],
            processor.get_mm_token_expansion_spec(processor._processor, first),
        )
        second_expansion = processor.mm_token_expansion(
            [VIDEO],
            processor.get_mm_token_expansion_spec(processor._processor, first),
        )
        self.assertEqual(first_expansion, second_expansion)
        self.assertEqual(list(metadata.frames_indices), indices)

    def test_audio_source_options_keep_truncation_and_batch_padding(self):
        processor = _make_hf_processor(
            Qwen3OmniMoeProcessor, 16, model_type="qwen3_omni_moe"
        )
        audio = np.zeros(3200, dtype=np.float32)
        audio_options = {"truncation": True, "max_length": 1600}
        first = processor.process_media(audios=[audio], **audio_options)
        second = processor.process_media(
            audios=[
                {"url": audio, "preprocess_kwargs": audio_options},
                audio,
            ],
            truncation=False,
        )
        torch.testing.assert_close(
            first["input_features"],
            second["input_features"][:1, ..., : first["input_features"].shape[-1]],
        )
        first_replacements = processor.get_mm_token_expansion_spec(
            processor._processor, first
        )
        mm_token_expansion_spec = processor.get_mm_token_expansion_spec(
            processor._processor, second
        )
        self.assertEqual(
            first_replacements[-1][1][0], mm_token_expansion_spec[-1][1][0]
        )
        self.assertLess(
            len(mm_token_expansion_spec[-1][1][0]),
            len(mm_token_expansion_spec[-1][1][1]),
        )
        self.assertIsInstance(second["input_features"], torch.Tensor)
        original_length = first["feature_attention_mask"].shape[-1]
        self.assertEqual(
            second["feature_attention_mask"][:1, original_length:]
            .count_nonzero()
            .item(),
            0,
        )
        expanded = processor.mm_token_expansion(
            [AUDIO, EOS, AUDIO], mm_token_expansion_spec
        )
        with patch.object(processor, "_build_position_inputs", return_value={}):
            serving = processor.sglang_post_process(
                expanded, second, mm_token_expansion_spec=mm_token_expansion_spec
            )
        torch.testing.assert_close(
            torch.cat([item.feature for item in serving.mm_items]),
            second["input_features"],
        )
        torch.testing.assert_close(
            torch.cat([item.feature_attention_mask for item in serving.mm_items]),
            second["feature_attention_mask"],
        )
        for item in serving.mm_items:
            self.assertEqual(
                item.feature.untyped_storage().data_ptr(),
                second["input_features"].untyped_storage().data_ptr(),
            )
            self.assertEqual(
                item.feature_attention_mask.untyped_storage().data_ptr(),
                second["feature_attention_mask"].untyped_storage().data_ptr(),
            )


if __name__ == "__main__":
    unittest.main()
