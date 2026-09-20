"""CPU contracts for TokenSpaceMultimodalProcessor (no model weights).

sources -> shared mixin loader + pools -> TokenSpace.process_media -> BatchFeature
                                                                       |
native BatchFeature -> mm_token_expansion_spec                          |
partly expanded IDs + boundary -> suffix matcher -> final IDs           |
                                                      |                |
                             Base.sglang_post_process <-----------------+
                                       |
                    collect -> full offsets -> serving split

raw media + IDs -> supported loader -> same IDs
               -> disabled loader  -> legacy text decode
loaded media + text -> disabled sync -> original HF fields, no serving build

The matcher preserves history and never scans inserted tokens. Serving offsets
retain adjacent and empty sources without mutating native tensors or metadata.
Loading preserves source options, wrappers and sample rates; processing uses HF
kwargs merging and runs off the event loop, including cloned workers.
"""

import base64
import concurrent.futures
import io
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import BatchFeature, ProcessorMixin

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.mm_token_expansion import expand_token_placeholders
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.multimodal.processors.token_space_processor import (
    TokenSpaceMultimodalProcessor,
)
from sglang.srt.utils.common import ImageData, VideoData
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMMTokenExpansion(unittest.TestCase):
    def test_preserves_non_media_ids_and_does_not_expand_insertions_again(self):
        original_input_ids = [71, 72, 99, 80, 98, 99, 81]
        expanded_input_ids = expand_token_placeholders(
            original_input_ids,
            [([99], [[99, 99], [99]]), ([98], [[98, 99, 98]])],
        )
        self.assertEqual(expanded_input_ids, [71, 72, 99, 99, 80, 98, 99, 98, 99, 81])
        self.assertEqual(original_input_ids, [71, 72, 99, 80, 98, 99, 81])
        self.assertEqual(
            expand_token_placeholders(original_input_ids, []), original_input_ids
        )

    def test_matches_whole_patterns_and_preserves_untouched_prefix(self):
        # [history | START PAD PAD END text] -> [history | fragment text insertion]
        # The text anchor is retained in its replacement; callers provide no positions.
        for history in ([], [70, 99, 99, 71]):
            with self.subTest(history=history):
                original = history + [80, 99, 99, 81, 72]
                media_fragments = ([[99, 99]] if history else []) + [[88, 99, 77]]
                self.assertEqual(
                    expand_token_placeholders(
                        original,
                        [
                            ([80, 99, 99, 81], media_fragments),
                            ([72], [[72, 98, 99, 98]]),
                        ],
                        mm_token_expansion_start_len=len(history),
                    ),
                    history + [88, 99, 77, 72, 98, 99, 98],
                )
                self.assertEqual(original, history + [80, 99, 99, 81, 72])

    def test_adjacent_patterns_follow_rule_order_without_overlapping(self):
        # [START PAD][START PAD][PAD] -> [fragment 0][fragment 1][bare PAD fragment]
        # Inner PADs and lower-priority START rules must not consume fragments.
        self.assertEqual(
            expand_token_placeholders(
                [80, 99, 80, 99, 99],
                [([80, 99], [[88], [89]]), ([80], []), ([99], [[77]])],
            ),
            [88, 89, 77],
        )
        # A bare START after a complete match can still use the lower-priority rule.
        self.assertEqual(
            expand_token_placeholders(
                [80, 99, 80],
                [([80, 99], [[88]]), ([80], [[77]])],
            ),
            [88, 77],
        )
        with self.assertRaisesRegex(ValueError, "patterns must not be empty"):
            expand_token_placeholders([80], [([], [[99]])])

    def test_offsets_include_spans_touching_both_ends(self):
        for input_ids, expected_offsets in [
            ([99, 99, 5, 99], [(0, 1), (3, 3)]),
            ([99, 99], [(0, 1)]),
            ([], []),
        ]:
            with self.subTest(input_ids=input_ids):
                self.assertEqual(
                    BaseMultimodalProcessor.get_mm_items_offset(
                        torch.tensor(input_ids), 99
                    ),
                    expected_offsets,
                )

    def test_validates_expansion_boundary_before_matching(self):
        input_ids = [10, 99]
        mm_token_expansion_spec = [([99], [[99, 99]])]
        for start in (-1, 3, 0.5):
            with (
                self.subTest(start=start),
                self.assertRaisesRegex(ValueError, "mm_token_expansion_start_len"),
            ):
                expand_token_placeholders(input_ids, mm_token_expansion_spec, start)
        self.assertEqual(
            expand_token_placeholders(
                input_ids, mm_token_expansion_spec, len(input_ids)
            ),
            input_ids,
        )

    def test_rejects_missing_or_surplus_media(self):
        for input_ids, mm_token_expansion_spec in [
            ([], [([99], [[99]])]),
            ([99], [([99], [])]),
            ([80, 99, 81], [([80, 99, 81], [])]),
            ([80, 99], [([80, 99, 81], [[99]])]),
        ]:
            with self.subTest(input_ids=input_ids), self.assertRaises(ValueError):
                expand_token_placeholders(input_ids, mm_token_expansion_spec)

    def test_expansion_suffix_uses_trailing_media_and_preserves_prefix(self):
        # One modality's expanded fragment can contain another modality's marker.
        history = [71, 99, 99, 72, 98, 99, 98, 73]
        historical_expansion_spec = [([99], [[99, 99]]), ([98], [[98, 99, 98]])]
        mm_token_expansion_spec = [
            ([99], [[99, 99], [99, 99, 99]]),
            ([98], [[98, 99, 98]]),
        ]
        for suffix, expected_suffix in [([99, 74], [99, 99, 99, 74]), ([74], [74])]:
            with self.subTest(suffix=suffix):
                original = history + suffix
                self.assertEqual(
                    expand_token_placeholders(
                        original,
                        mm_token_expansion_spec
                        if 99 in suffix
                        else historical_expansion_spec,
                        mm_token_expansion_start_len=len(history),
                    ),
                    history + expected_suffix,
                )
        with self.assertRaises(ValueError):
            expand_token_placeholders(
                history + [98, 98], mm_token_expansion_spec, len(history)
            )


class _FakeMediaProcessor(ProcessorMixin):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)


class _TokenExpansionProcessor(TokenSpaceMultimodalProcessor, BaseMultimodalProcessor):
    process_mm_data_async = BaseMultimodalProcessor.process_token_space_mm_data_async
    keep_mm_features_on_device = False
    precompute_hash_before_cpu_transfer = False

    def __init__(self):
        self._tokenizer = SimpleNamespace(
            decode=Mock(side_effect=AssertionError("Caller IDs must never decode")),
            encode=Mock(return_value=[10, 99, 99, 11]),
            bos_token="<bos>",
            init_kwargs={},
        )
        self._processor = _FakeMediaProcessor(self._tokenizer)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>", image_token_id=99
        )
        self.image_config = self.video_config = self.audio_config = {}
        self.disable_fast_image_processor = True
        self.mm_processor_executor = None
        self.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._cpu_executor_lock = threading.Lock()
        self.processing_threads = []

    def process_images(self, images, processor, **kwargs):
        self.processing_threads.append(threading.get_ident())
        token_counts = [image.width for image in images]
        pixels = torch.tensor(token_counts, dtype=torch.float32).unsqueeze(1)
        return {"pixel_values": pixels, "num_image_tokens": token_counts}

    def get_mm_token_expansion_spec(self, processor, media_features):
        return [([99], [[99] * count for count in media_features["num_image_tokens"]])]

    def _temporary_fast_processor_cuda_pool(self, device):
        return nullcontext()

    def _finalize_mm_items(self, mm_items, *, images):
        return mm_items

    def collect_mm_items_from_processor_output(self, media_features, modality=None):
        return [
            MultimodalDataItem(modality=Modality.IMAGE, feature=pixels.unsqueeze(0))
            for pixels in media_features["pixel_values"]
        ]


class TestTokenSpaceProcessor(unittest.IsolatedAsyncioTestCase):
    def make_processor(self):
        processor = _TokenExpansionProcessor()
        self.addCleanup(processor.io_executor.shutdown)
        return processor

    async def test_supported_loaders_preserve_ids_and_source_options(self):
        processor = self.make_processor()
        image_bytes = io.BytesIO()
        Image.new("RGBA", (3, 2), (10, 20, 30, 40)).save(image_bytes, format="PNG")
        image_url = (
            "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()
        )
        video_decoder = object()
        image_options = {"do_resize": False}
        video_options = {"frame_indices": [0, 4]}
        for loader in (processor.load_mm_data, processor.fast_load_mm_data):
            for input_ids, return_text in (
                (None, True),
                ([], False),
                ([10, 99, 99, 11, 99], True),
            ):
                with (
                    self.subTest(loader=loader.__name__, input_ids=input_ids),
                    patch(
                        "sglang.srt.multimodal.media_processor.load_video",
                        return_value=video_decoder,
                    ) as load_video,
                ):
                    loaded = await loader(
                        prompt=input_ids,
                        multimodal_tokens=None,
                        return_text=return_text,
                        image_data=[
                            ImageData(image_url, preprocess_kwargs=image_options)
                        ],
                        video_data=[
                            VideoData(
                                "/shared/video.mp4", preprocess_kwargs=video_options
                            )
                        ],
                    )
                    self.assertIs(loaded.input_ids, input_ids)
                    self.assertEqual(loaded.input_text, "")
                    image = loaded.images[0]["url"]
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
                    self.assertEqual(
                        loaded.images[0]["preprocess_kwargs"], image_options
                    )
                    self.assertIs(loaded.videos[0]["url"], video_decoder)
                    self.assertEqual(
                        loaded.videos[0]["preprocess_kwargs"], video_options
                    )
                    load_video.assert_called_once_with(
                        VideoData("/shared/video.mp4", preprocess_kwargs=video_options),
                        None,
                    )
        processor._tokenizer.decode.assert_not_called()

    async def test_loader_resamples_audio_before_processing(self):
        processor = self.make_processor()
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, np.ones(480, dtype=np.float32), 48000, format="WAV")
        for input_ids in (None, [10, 98, 11]):
            with self.subTest(input_ids=input_ids):
                loaded = await processor.load_mm_data(
                    prompt=input_ids,
                    audio_data=[
                        audio_bytes.getvalue(),
                        {
                            "url": audio_bytes.getvalue(),
                            "preprocess_kwargs": {"sampling_rate": 32000},
                        },
                    ],
                )
                self.assertIs(loaded.input_ids, input_ids)
                self.assertEqual(loaded.audios[0].shape, (160,))
                self.assertEqual(loaded.audios[1]["url"].shape, (320,))
        processor._tokenizer.decode.assert_not_called()

    async def test_loaded_media_is_reusable_without_token_processing(self):
        processor = self.make_processor()
        video_decoder = object()
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, np.ones(160, dtype=np.float32), 16000, format="WAV")
        with patch(
            "sglang.srt.multimodal.media_processor.load_video",
            return_value=video_decoder,
        ):
            loaded = await processor.fast_load_mm_data(
                prompt=None,
                multimodal_tokens=None,
                image_data=[Image.new("RGB", (width, 1)) for width in (2, 3)],
                video_data=["/shared/video.mp4"],
                audio_data=[audio_bytes.getvalue()],
                audio_sample_rate=16000,
            )
        video_features = {
            "pixel_values_videos": torch.ones(4, 3),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "video_metadata": [{"fps": 2, "frames_indices": [0, 2]}],
        }
        audio_features = {
            "input_features": torch.ones(1, 4, 8),
            "feature_attention_mask": torch.ones(1, 8, dtype=torch.long),
        }
        with (
            patch.object(
                processor,
                "process_videos",
                return_value=video_features,
            ) as process_videos,
            patch.object(
                processor,
                "process_audio",
                return_value=audio_features,
            ) as process_audio,
            patch.multiple(
                processor,
                get_mm_token_expansion_spec=Mock(side_effect=AssertionError),
                mm_token_expansion=Mock(side_effect=AssertionError),
            ),
        ):
            features = processor.process_media(
                images=loaded.images, videos=loaded.videos, audios=loaded.audios
            )
        self.assertIsInstance(features, BatchFeature)
        self.assertEqual(features["pixel_values"].tolist(), [[2.0], [3.0]])
        self.assertIs(process_videos.call_args.args[0][0], video_decoder)
        self.assertIs(process_audio.call_args.args[0][0], loaded.audios[0])
        for name, value in (video_features | audio_features).items():
            self.assertIs(features[name], value)
        processor._tokenizer.encode.assert_not_called()
        processor._tokenizer.decode.assert_not_called()

    async def test_disabled_loaders_keep_legacy_text_decoding(self):
        processor = self.make_processor()
        processor.use_token_space_processor = False
        processor.skip_tokenizer_init = False
        processor.mm_tokens.build(processor._processor)
        processor._tokenizer.decode = Mock(return_value="<image>")
        input_ids = [10, 99, 11]
        image = Image.new("RGB", (3, 2))
        for loader in (processor.load_mm_data, processor.fast_load_mm_data):
            with self.subTest(loader=loader.__name__):
                kwargs = (
                    {"input_ids": input_ids}
                    if loader == processor.fast_load_mm_data
                    else {}
                )
                loaded = await loader(
                    prompt=input_ids,
                    multimodal_tokens=processor.mm_tokens,
                    image_data=[image],
                    **kwargs,
                )
                self.assertEqual(loaded.input_text, "<image>")
                self.assertIs(loaded.input_ids, input_ids)
                self.assertEqual(loaded.images, [image])
                processor._tokenizer.decode.assert_called_once_with(input_ids)
                processor._tokenizer.decode.reset_mock()

    async def test_preprocessed_ids_do_not_resolve_audio_sample_rate(self):
        processor = self.make_processor()
        del processor._processor.feature_extractor
        audio = {"format": "processor_output", "input_features": [1.0]}
        input_ids = [10, 98, 11]
        loaded = await processor.load_mm_data(prompt=input_ids, audio_data=[audio])
        self.assertIs(loaded.input_ids, input_ids)
        self.assertIs(loaded.audios[0], audio)
        processor._tokenizer.decode.assert_not_called()

    async def test_async_workers_preserve_history_and_do_not_block_event_loop(self):
        # [2-token image | new placeholder] -> two distinct adjacent media slots.
        for use_workers in (False, True):
            with self.subTest(use_workers=use_workers):
                processor = self.make_processor()
                if use_workers:
                    processor.mm_processor_executor = MultimodalProcessorExecutor(
                        lambda: processor._processor, 1
                    )
                    self.addCleanup(processor.mm_processor_executor.shutdown)
                result = await processor.process_mm_data_async(
                    input_ids=[10, 99, 99, 99, 11],
                    image_data=[Image.new("RGB", (width, 1)) for width in (2, 3)],
                    mm_token_expansion_start_len=3,
                )
                self.assertEqual(result.input_ids, [10, 99, 99, 99, 99, 99, 11])
                self.assertEqual(
                    [item.offsets for item in result.mm_items], [[(1, 2)], [(3, 5)]]
                )
                self.assertTrue(
                    all(
                        worker != threading.get_ident()
                        for worker in processor.processing_threads
                    )
                )
                processor._tokenizer.decode.assert_not_called()

    def test_text_and_ids_share_the_same_processing_pipeline(self):
        processor = self.make_processor()
        ids_result = processor.process_mm_data(
            input_ids=[10, 99, 99, 11],
            images=[Image.new("RGB", (width, 1)) for width in (2, 3)],
        )
        text_result = processor.process_mm_data(
            "<bos> rendered text",
            images=[Image.new("RGB", (width, 1)) for width in (2, 3)],
        )
        self.assertEqual(ids_result.input_ids, text_result.input_ids)
        for ids_item, text_item in zip(ids_result.mm_items, text_result.mm_items):
            torch.testing.assert_close(ids_item.feature, text_item.feature)
            self.assertEqual(ids_item.offsets, text_item.offsets)
        processor._tokenizer.encode.assert_called_once_with(
            "<bos> rendered text", add_special_tokens=False
        )
        processor._tokenizer.decode.assert_not_called()

    def test_disabled_sync_preserves_the_legacy_processor_output(self):
        processor = self.make_processor()
        processor.use_token_space_processor = False
        processor._tokenizer_auto_adds_specials = False
        processor.FEATURE_NAMES = ["pixel_values"]
        images = [Image.new("RGB", (2, 1))]
        native_output = BatchFeature(
            {
                "input_ids": torch.tensor([[10, 99, 99, 11]]),
                "pixel_values": torch.ones(1, 2),
            }
        )
        hf_processor = Mock(tokenizer=processor._tokenizer, return_value=native_output)
        with patch.object(processor, "process_media", side_effect=AssertionError):
            output = processor.process_mm_data(
                "image prompt", images=images, processor=hf_processor
            )
        self.assertIs(output, native_output)
        hf_processor.assert_called_once_with(
            text=["image prompt"], images=images, padding=True, return_tensors="pt"
        )
        processor._tokenizer.encode.assert_not_called()

    async def test_async_native_media_does_not_assemble_language_model_inputs(self):
        processor = self.make_processor()
        with patch.multiple(
            processor,
            get_mm_token_expansion_spec=Mock(side_effect=AssertionError),
            mm_token_expansion=Mock(side_effect=AssertionError),
            sglang_post_process=Mock(side_effect=AssertionError),
        ):
            output = await processor.process_media_async(
                images=[Image.new("RGB", (width, 1)) for width in (2, 3)],
            )
        self.assertIsInstance(output, BatchFeature)
        self.assertEqual(set(output), {"pixel_values", "num_image_tokens"})
        self.assertEqual(output["pixel_values"].tolist(), [[2.0], [3.0]])
        self.assertNotEqual(processor.processing_threads[0], threading.get_ident())
        processor._tokenizer.encode.assert_not_called()
        processor._tokenizer.decode.assert_not_called()

    def test_source_offsets_preserve_empty_items_and_reject_split_segments(self):
        processor = self.make_processor()
        for widths, expected_offsets in (
            ((0, 2), [[], [(0, 1)]]),
            ((2, 0, 1), [[(0, 1)], [], [(2, 2)]]),
            ((0, 0), [[], []]),
        ):
            with self.subTest(widths=widths):
                media = processor.process_media(
                    images=[Image.new("RGB", (width, 1)) for width in widths]
                )
                expanded = processor.mm_token_expansion(
                    [99] * len(widths),
                    processor.get_mm_token_expansion_spec(processor._processor, media),
                )
                serving = processor.sglang_post_process(expanded, media)
                self.assertEqual(expanded, [99] * sum(widths))
                self.assertEqual(
                    [item.offsets for item in serving.mm_items], expected_offsets
                )
        media = processor.process_media(
            images=[Image.new("RGB", (width, 1)) for width in (0, 2)]
        )
        with self.assertRaisesRegex(ValueError, "token span"):
            processor.sglang_post_process([99, 7, 99], media)

    def test_serving_postprocessing_preserves_reusable_media_and_full_offsets(self):
        processor = self.make_processor()
        processor.image_config = {"crop": False}
        image_kwargs = {"size": 2}
        media = processor.process_media(
            images=[Image.new("RGB", (width, 1)) for width in (2, 3)],
            images_kwargs=image_kwargs,
        )
        self.assertEqual(image_kwargs, {"size": 2})
        pixels = media["pixel_values"]
        before = pixels.clone()
        for input_ids, boundary in (([99, 99], 0), ([99, 99, 99], 2), ([99] * 5, 5)):
            expanded = processor.mm_token_expansion(
                input_ids,
                processor.get_mm_token_expansion_spec(processor._processor, media),
                boundary,
            )
            serving = processor.sglang_post_process(expanded, media)
            self.assertEqual(serving.input_ids, [99] * 5)
            self.assertEqual(
                [item.offsets for item in serving.mm_items], [[(0, 1)], [(2, 4)]]
            )
            torch.testing.assert_close(pixels, before)
            self.assertEqual(set(media), {"pixel_values", "num_image_tokens"})


if __name__ == "__main__":
    unittest.main()
