"""Unit tests for ``BaseMultimodalProcessor._load_single_item`` image decoding.

Regression test for the change that forces the (otherwise lazy) PIL decode inside
``_load_single_item`` — which runs in the ``io_executor`` worker thread — instead of
letting it fire lazily on the main event-loop thread later (inside
``pil_to_tensor``/``tobytes`` during processing). The behavior of the returned image
(mode, pixels) must be unchanged; only *when/where* the decode happens differs.

No server, no model loading — pure CPU.

source -> Base._load_single_item -> decoded image or client/server error
             IO workers         -> ordered media + per-source options
ImageData / VideoData wrappers -> decoder-owned unwrapping
JPEG + Modality.IMAGE -> shared mixin -> Base / TokenSpace -> same decoder policy
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import asyncio
import concurrent.futures
import io
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.token_space_processor import (
    TokenSpaceMultimodalProcessor,
)
from sglang.srt.utils import common
from sglang.srt.utils.nvjpeg_decoder import _NvJpegDecoderPool
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    # gpu_image_decode=False forces the PIL (CPU) path so the test needs no GPU and
    # exercises exactly the lazy-decode branch the fix targets. The abstract methods
    # are never called: we only invoke the _load_single_item classmethod.
    gpu_image_decode = False

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _png_bytes(mode: str = "RGB", size=(8, 8)) -> bytes:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    img = Image.fromarray(arr, "RGB").convert(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(8, 8)) -> bytes:
    arr = (np.random.RandomState(0).rand(size[1], size[0], 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=90, subsampling=2)
    return buf.getvalue()


def _is_decoded(img: Image.Image) -> bool:
    """A lazily-opened PIL image has no decoded core yet; ``load()`` populates it.
    PIL's ``.im`` property requires a completed load and raises otherwise."""
    try:
        return img.im is not None
    except Exception:
        return False


class TestLoadSingleItemImageDecode(CustomTestCase):
    def test_plain_open_is_lazy(self):
        # Documents why the fix matters: a bare Image.open is not decoded yet, so
        # without the fix the decode would land on the caller (main) thread.
        lazy = Image.open(io.BytesIO(_png_bytes()))
        self.assertFalse(_is_decoded(lazy))

    def test_load_single_item_forces_decode(self):
        img = _StubProcessor._load_single_item(_png_bytes("RGB"), Modality.IMAGE)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        # The fix: decode is forced inside _load_single_item, not lazily later.
        self.assertTrue(_is_decoded(img))

    def test_rgba_converted_to_rgb_and_decoded(self):
        img = _StubProcessor._load_single_item(_png_bytes("RGBA"), Modality.IMAGE)
        # Existing alpha-discard behavior preserved.
        self.assertEqual(img.mode, "RGB")
        self.assertTrue(_is_decoded(img))

    def test_pixels_match_reference(self):
        # Output must be bit-identical to the pre-fix path (open -> [convert]).
        data = _png_bytes("RGB")
        img = _StubProcessor._load_single_item(data, Modality.IMAGE)
        ref = Image.open(io.BytesIO(data)).convert("RGB")
        np.testing.assert_array_equal(np.asarray(img), np.asarray(ref))

    def test_fast_loader_preserves_invalid_input_as_value_error(self):
        processor = object.__new__(_StubProcessor)
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
            patch.object(
                _StubProcessor,
                "_load_single_item",
                side_effect=ValueError("invalid base64 image"),
            ),
        ):
            processor.io_executor = executor
            with self.assertRaisesRegex(ValueError, "invalid base64 image"):
                asyncio.run(
                    processor.fast_load_mm_data(
                        prompt="<image>",
                        multimodal_tokens=None,
                        image_data=["bad-image"],
                    )
                )

    def test_fast_loader_submits_all_media_and_preserves_source_options(self):
        all_started = threading.Barrier(4, timeout=5)
        calls = []

        def load(data, modality, frame_count_limit, sample_rate, discard_alpha):
            calls.append(
                (data, modality, frame_count_limit, sample_rate, discard_alpha)
            )
            all_started.wait()
            source = (
                data.url
                if isinstance(data, (common.ImageData, common.VideoData))
                else data
            )
            return f"decoded:{source}"

        sources = {
            "image_data": [
                common.ImageData(url="first", preprocess_kwargs={"size": 10}),
                {"url": "second", "preprocess_kwargs": {"size": 20}},
            ],
            "video_data": [common.VideoData(url="video")],
            "audio_data": [
                {"url": "audio", "preprocess_kwargs": {"sampling_rate": 8000}}
            ],
        }
        processor = object.__new__(_StubProcessor)
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor,
            patch.object(_StubProcessor, "_load_single_item", side_effect=load),
        ):
            processor.io_executor = executor
            loaded = asyncio.run(
                processor.fast_load_mm_data(
                    prompt=None,
                    multimodal_tokens=None,
                    **sources,
                    audio_sample_rate=16000,
                    discard_alpha_channel=False,
                )
            )
        self.assertEqual(
            loaded.images,
            [
                {"url": "decoded:first", "preprocess_kwargs": {"size": 10}},
                {"url": "decoded:second", "preprocess_kwargs": {"size": 20}},
            ],
        )
        self.assertEqual(loaded.videos[0]["url"], "decoded:video")
        self.assertEqual(loaded.audios[0]["url"], "decoded:audio")
        self.assertCountEqual(
            calls,
            [
                (sources["image_data"][0], Modality.IMAGE, None, 16000, False),
                ("second", Modality.IMAGE, None, 16000, False),
                (sources["video_data"][0], Modality.VIDEO, None, 16000, False),
                ("audio", Modality.AUDIO, None, 8000, False),
            ],
        )

    def test_unreachable_image_url_is_a_client_error(self):
        with patch(
            "sglang.srt.multimodal.media_processor.load_image",
            side_effect=requests.ConnectionError("connection refused"),
        ):
            with self.assertRaisesRegex(ValueError, "connection refused"):
                _StubProcessor._load_single_item(
                    "https://127.0.0.1:1/not-an-image.png", Modality.IMAGE
                )

    def test_invalid_image_bytes_are_a_client_error(self):
        with self.assertRaisesRegex(ValueError, "cannot identify image file"):
            _StubProcessor._load_single_item(b"not an image", Modality.IMAGE)

    def test_unexpected_loader_bug_remains_a_server_error(self):
        with patch(
            "sglang.srt.multimodal.media_processor.load_image",
            side_effect=TypeError("unexpected loader bug"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected loader bug"):
                _StubProcessor._load_single_item(b"image", Modality.IMAGE)

    def test_shared_and_serving_choose_same_jpeg_decoder(self):
        jpeg = _jpeg_bytes()
        decoded = torch.zeros((3, 8, 8), dtype=torch.uint8)
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch.object(common, "decode_jpeg", return_value=decoded) as decode,
        ):
            for processor in (BaseMultimodalProcessor, TokenSpaceMultimodalProcessor):
                self.assertIs(
                    processor._load_single_item(jpeg, Modality.IMAGE), decoded
                )
        self.assertEqual(decode.call_count, 2)

    def test_high_fidelity_gpu_jpeg_decoder_is_selected(self):
        data = _jpeg_bytes()
        expected = torch.zeros((3, 8, 8), dtype=torch.uint8)
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch(
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
                return_value=expected,
            ) as decode,
        ):
            image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")

        self.assertIs(image, expected)
        decode.assert_called_once_with(data)

    def test_high_fidelity_gpu_jpeg_decoder_falls_back_to_pil(self):
        data = _jpeg_bytes()
        common._warn_fancy_jpeg_fallback.cache_clear()
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch(
                "sglang.srt.utils.nvjpeg_decoder.decode_jpeg_with_fancy_upsampling",
                side_effect=ImportError("nvImageCodec is unavailable"),
            ),
        ):
            image, _ = common.load_image(data, gpu_image_decode="nvjpeg_fancy")

        self.assertIsInstance(image, Image.Image)
        reference = Image.open(io.BytesIO(data))
        np.testing.assert_array_equal(np.asarray(image), np.asarray(reference))

    def test_high_fidelity_decoder_uses_fancy_planar_rgb_and_reuses_pool(self):
        expected = torch.zeros((3, 8, 8), dtype=torch.uint8)
        fake_format = object()

        class FakeImage:
            def to_dlpack(self, *, cuda_stream):
                self.cuda_stream = cuda_stream
                return object()

        class FakeDecoder:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.instances.append(self)

            def decode(self, data, *, params, cuda_stream):
                self.call = (data, params, cuda_stream)
                return FakeImage()

        class FakeDecodeParams:
            def __init__(self, *, sample_format, apply_exif_orientation):
                self.sample_format = sample_format
                self.apply_exif_orientation = apply_exif_orientation

        fake_codec = SimpleNamespace(
            DecodeParams=FakeDecodeParams,
            Decoder=FakeDecoder,
            SampleFormat=SimpleNamespace(P_RGB=fake_format),
        )
        nvidia = types.ModuleType("nvidia")
        nvidia.nvimgcodec = fake_codec

        with (
            patch.dict(sys.modules, {"nvidia": nvidia}),
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=7),
            ),
            patch.object(torch, "from_dlpack", return_value=expected),
        ):
            pool = _NvJpegDecoderPool(device_id=2)
            self.assertIs(pool.decode(b"jpeg"), expected)
            self.assertIs(pool.decode(b"jpeg"), expected)

        self.assertEqual(len(FakeDecoder.instances), 1)
        decoder = FakeDecoder.instances[0]
        self.assertEqual(decoder.kwargs["device_id"], 2)
        self.assertEqual(decoder.kwargs["max_num_cpu_threads"], 1)
        self.assertIn(":fancy_upsampling=1", decoder.kwargs["options"])
        self.assertIs(pool._decode_params.sample_format, fake_format)
        self.assertFalse(pool._decode_params.apply_exif_orientation)


if __name__ == "__main__":
    unittest.main()
