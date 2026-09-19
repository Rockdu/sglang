"""CPU contracts for the shared media pipeline (no model weights).

sources -> fast loader (decode, no tokens) -> process_media -> tensors + metadata
                                                        |
partly expanded IDs + A -> suffix matcher -> final IDs
                                                        |
                           build -> training dict / serving items + full offsets

Text tokenizes once before the same pipeline; IDs never decode. Both single
worker and cloned workers run off the event loop. Adjacent media retain separate
source offsets, empty sources consume no span, and segments cannot cross gaps.
Repeated build calls cannot mutate the media result.
Native training builds neither serving offsets nor replacement fragments.
Loading reuses common decoders and preserves per-source processing options.
Audio resampling precedes processing and honors per-source sample rates.
IDs-only matching retains its tuple-list contract and rejects bad counts.
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

from sglang.srt.multimodal.media_processing import (
    MediaProcessOutput,
    ProcessedMediaItem,
)
from sglang.srt.multimodal.mm_token_expansion import expand_token_placeholders
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
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
        replacements = [([99], [[99, 99]])]
        for start in (-1, 3, 0.5):
            with (
                self.subTest(start=start),
                self.assertRaisesRegex(ValueError, "mm_token_expansion_start_len"),
            ):
                expand_token_placeholders(input_ids, replacements, start)
        self.assertEqual(
            expand_token_placeholders(input_ids, replacements, len(input_ids)),
            input_ids,
        )

    def test_rejects_missing_or_surplus_media(self):
        for input_ids, replacements in [
            ([], [([99], [[99]])]),
            ([99], [([99], [])]),
            ([80, 99, 81], [([80, 99, 81], [])]),
            ([80, 99], [([80, 99, 81], [[99]])]),
        ]:
            with self.subTest(input_ids=input_ids), self.assertRaises(ValueError):
                expand_token_placeholders(input_ids, replacements)

    def test_expansion_suffix_uses_trailing_media_and_preserves_prefix(self):
        # One modality's expanded fragment can contain another modality's marker.
        history = [71, 99, 99, 72, 98, 99, 98, 73]
        historical_replacements = [([99], [[99, 99]]), ([98], [[98, 99, 98]])]
        replacements = [([99], [[99, 99], [99, 99, 99]]), ([98], [[98, 99, 98]])]
        for suffix, expected_suffix in [([99, 74], [99, 99, 99, 74]), ([74], [74])]:
            with self.subTest(suffix=suffix):
                original = history + suffix
                self.assertEqual(
                    expand_token_placeholders(
                        original,
                        replacements if 99 in suffix else historical_replacements,
                        mm_token_expansion_start_len=len(history),
                    ),
                    history + expected_suffix,
                )
        with self.assertRaises(ValueError):
            expand_token_placeholders(history + [98, 98], replacements, len(history))


class _TokenExpansionProcessor(BaseMultimodalProcessor):
    supports_token_expansion = True
    uses_hf_processor_kwargs = False
    keep_mm_features_on_device = False
    precompute_hash_before_cpu_transfer = False

    def __init__(self):
        self._tokenizer = SimpleNamespace(
            decode=Mock(side_effect=AssertionError("Caller IDs must never decode")),
            encode=Mock(return_value=[10, 99, 99, 11]),
            bos_token="<bos>",
        )
        self._processor = SimpleNamespace(tokenizer=self._tokenizer)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>", image_token_id=99
        )
        self.image_config = self.video_config = self.audio_config = {}
        self.disable_fast_image_processor = True
        self.mm_processor_executor = None
        self.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._cpu_executor_lock = threading.Lock()
        self.processing_threads = []

    def _temporary_fast_processor_cuda_pool(self, device):
        return nullcontext()

    def _finalize_mm_items(self, mm_items, *, images):
        return mm_items

    def process_images(self, images, processor, **kwargs):
        self.processing_threads.append(threading.get_ident())
        token_counts = [image.width for image in images]
        pixels = torch.tensor(token_counts, dtype=torch.float32).unsqueeze(1)
        return MediaProcessOutput(
            {"pixel_values": pixels},
            [
                ProcessedMediaItem(
                    media_id=("image", index),
                    encoder_inputs={"pixel_values": pixels[index : index + 1]},
                    metadata={"token_count": int(count)},
                    feature_name="pixel_values",
                )
                for index, count in enumerate(token_counts)
            ],
        )

    def get_mm_token_replacements(self, processor, processed_media):
        return [
            (
                [99],
                [
                    [([99] * item.metadata["token_count"], item.media_id)]
                    for item in processed_media["image"].items
                ],
            )
        ]


class TestBaseTokenExpansion(unittest.IsolatedAsyncioTestCase):
    def make_processor(self):
        processor = _TokenExpansionProcessor()
        self.addCleanup(processor.io_executor.shutdown)
        return processor

    async def test_promptless_loader_decodes_media_and_preserves_options(self):
        processor = self.make_processor()
        image_bytes = io.BytesIO()
        Image.new("RGBA", (3, 2), (10, 20, 30, 40)).save(image_bytes, format="PNG")
        image_url = (
            "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode()
        )
        video_decoder = object()
        image_options = {"do_resize": False}
        video_options = {"frame_indices": [0, 4]}
        with patch(
            "sglang.srt.multimodal.processors.base_processor.load_video",
            return_value=video_decoder,
        ) as load_video:
            loaded = await processor.load_mm_data(
                image_data=[ImageData(image_url, preprocess_kwargs=image_options)],
                video_data=[
                    VideoData("/shared/video.mp4", preprocess_kwargs=video_options)
                ],
            )
        image = loaded.images[0]["url"]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(loaded.images[0]["preprocess_kwargs"], image_options)
        self.assertIs(loaded.videos[0]["url"], video_decoder)
        self.assertEqual(loaded.videos[0]["preprocess_kwargs"], video_options)
        load_video.assert_called_once_with("/shared/video.mp4", None)
        processor._tokenizer.decode.assert_not_called()

    async def test_loader_resamples_audio_before_processing(self):
        processor = self.make_processor()
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, np.ones(480, dtype=np.float32), 48000, format="WAV")
        loaded = await processor.load_mm_data(
            audio_data=[
                audio_bytes.getvalue(),
                {
                    "url": audio_bytes.getvalue(),
                    "preprocess_kwargs": {"sampling_rate": 32000},
                },
            ],
            audio_sample_rate=16000,
        )
        self.assertEqual(loaded.audios[0].shape, (160,))
        self.assertEqual(loaded.audios[1]["url"].shape, (320,))
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
        torch.testing.assert_close(ids_result["input_ids"], text_result["input_ids"])
        torch.testing.assert_close(
            ids_result["pixel_values"], text_result["pixel_values"]
        )
        processor._tokenizer.encode.assert_called_once_with(
            "<bos> rendered text", add_special_tokens=False
        )
        processor._tokenizer.decode.assert_not_called()

    async def test_async_training_consumer_preserves_explicit_options(self):
        processor = self.make_processor()
        output = await processor.process_mm_data_async(
            input_ids=[99, 99],
            image_data=[Image.new("RGB", (width, 1)) for width in (2, 3)],
            consumer="training",
            return_mm_token_type_ids=True,
        )
        self.assertEqual(output["input_ids"].tolist(), [[99] * 5])
        self.assertEqual(output["mm_token_type_ids"].tolist(), [[1] * 5])

    def test_source_offsets_preserve_empty_items_and_reject_split_segments(self):
        processor = self.make_processor()
        media = processor.process_media(
            images=[Image.new("RGB", (width, 1)) for width in (0, 2)]
        )
        # The first placeholder disappears; the second owns one two-token span.
        expanded = processor.mm_token_expansion([99, 99], media)
        serving = processor.build_multimodal_inputs(expanded, media)
        self.assertEqual(expanded, [99, 99])
        self.assertEqual([item.offsets for item in serving.mm_items], [[], [(0, 1)]])
        with self.assertRaisesRegex(ValueError, "token span"):
            processor.build_multimodal_inputs([99, 7, 99], media)

    def test_reusable_media_build_has_full_offsets_without_mutation(self):
        processor = self.make_processor()
        processor.image_config = {"crop": False}
        image_kwargs = {"size": 2}
        media = processor.process_media(
            images=[Image.new("RGB", (width, 1)) for width in (2, 3)],
            images_kwargs=image_kwargs,
        )
        self.assertEqual(image_kwargs, {"size": 2})
        pixels = media["image"].encoder_inputs["pixel_values"]
        before = pixels.clone()
        for input_ids, boundary in (([99, 99], 0), ([99, 99, 99], 2), ([99] * 5, 5)):
            expanded = processor.mm_token_expansion(input_ids, media, boundary)
            with (
                patch.object(
                    processor, "get_mm_item_offsets", side_effect=AssertionError
                ),
                patch.object(
                    processor, "get_mm_token_replacements", side_effect=AssertionError
                ),
            ):
                train = processor.build_multimodal_inputs(
                    expanded, media, consumer="training"
                )
            serving = processor.build_multimodal_inputs(expanded, media)
            self.assertIs(train["pixel_values"], pixels)
            self.assertEqual(train["input_ids"].tolist(), [[99] * 5])
            self.assertEqual(
                [item.offsets for item in serving.mm_items], [[(0, 1)], [(2, 4)]]
            )
            torch.testing.assert_close(pixels, before)
            self.assertNotIn("offsets", media["image"].items[0].metadata)


if __name__ == "__main__":
    unittest.main()
