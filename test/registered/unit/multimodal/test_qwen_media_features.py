"""Per-modality merges batch per-item native fields in item order.

Images and videos use the default concatenation; Qwen audio zero-pads.

    per-item native fields                   assembled native fields
    pixels [item patches, channels] -------> pixels: concatenated in item order
    grid [1, 3] ---------------------------> grid [items, 3]
    video metadata / times [1] ------------> metadata / times [items]
    audio [1, bins, T_i], mask [1, T_i] ---> audio [items, bins, max T_i],
                                             masks zero-padded to max T_i

Assembly keeps item order and every byte of each item, and never mutates inputs.
"""

import copy
import unittest

import torch
from transformers.video_utils import VideoMetadata

from sglang.srt.multimodal.token_space.qwen_vl import QwenTokenSpaceProcessStrategy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwenMediaFeatures(unittest.TestCase):
    def setUp(self):
        self.processor = object.__new__(QwenTokenSpaceProcessStrategy)

    def assert_features_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name, value in expected.items():
            with self.subTest(field=name):
                if isinstance(value, torch.Tensor):
                    self.assertEqual(actual[name].shape, value.shape)
                    self.assertEqual(actual[name].dtype, value.dtype)
                    self.assertEqual(
                        actual[name].contiguous().view(torch.uint8).numpy().tobytes(),
                        value.contiguous().view(torch.uint8).numpy().tobytes(),
                    )
                else:
                    self.assertEqual(actual[name], value)

    def test_image_items_concatenate_unequal_patches_in_order(self):
        items = [
            {
                "pixel_values": torch.arange(patches * 3, dtype=torch.float32).reshape(
                    patches, 3
                )
                + offset,
                "image_grid_thw": torch.tensor([grid]),
            }
            for patches, grid, offset in (
                (4, [1, 2, 2], 0),
                (8, [1, 2, 4], 100),
                (16, [1, 4, 4], 200),
            )
        ]
        items[0]["pixel_values"][0, 0] = -0.0
        items[1]["pixel_values"][1, 0] = float("nan")
        original = copy.deepcopy(items)

        merged = self.processor.merge_image_features(items)

        self.assert_features_equal(
            merged,
            {
                "pixel_values": torch.cat([item["pixel_values"] for item in original]),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4], [1, 4, 4]]),
            },
        )
        for item, before in zip(items, original):
            self.assert_features_equal(item, before)

    def test_video_metadata_and_times_follow_their_items(self):
        for time_name in (None, "second_per_grid_ts", "video_second_per_grid"):
            with self.subTest(time_name=time_name):
                items = []
                for patches, grid, metadata, seconds in (
                    (
                        8,
                        [2, 2, 2],
                        VideoMetadata(20, fps=8.0, frames_indices=[0, 3, 8, 9]),
                        0.5,
                    ),
                    (
                        24,
                        [3, 2, 4],
                        VideoMetadata(
                            24, fps=12.0, frames_indices=[0, 4, 8, 12, 16, 20]
                        ),
                        0.25,
                    ),
                    (
                        16,
                        [1, 4, 4],
                        VideoMetadata(16, fps=4.0, frames_indices=[0, 8]),
                        1.0,
                    ),
                ):
                    item = {
                        "pixel_values_videos": torch.full((patches, 3), float(patches)),
                        "video_grid_thw": torch.tensor([grid]),
                        "video_metadata": [metadata],
                    }
                    if time_name is not None:
                        item[time_name] = [seconds]
                    items.append(item)
                original = copy.deepcopy(items)

                ordered = [items[2], items[0], items[1]]
                merged = self.processor.merge_video_features(ordered)

                expected = {
                    "pixel_values_videos": torch.cat(
                        [item["pixel_values_videos"] for item in ordered]
                    ),
                    "video_grid_thw": torch.tensor([[1, 4, 4], [2, 2, 2], [3, 2, 4]]),
                    "video_metadata": [item["video_metadata"][0] for item in ordered],
                }
                if time_name is not None:
                    expected[time_name] = [1.0, 0.5, 0.25]
                self.assert_features_equal(merged, expected)
                for item, before in zip(items, original):
                    self.assert_features_equal(item, before)

    def test_audio_merge_zero_pads_unequal_widths_without_mutating_inputs(self):
        sources = [
            {
                "input_features": torch.arange(3 * width, dtype=torch.float16).reshape(
                    1, 3, width
                )
                + 1,
                "feature_attention_mask": torch.tensor(
                    [[1] * active + [0] * (width - active)]
                ),
            }
            for width, active in ((4, 2), (7, 0), (10, 8))
        ]
        original = copy.deepcopy(sources)
        expected = {
            "input_features": torch.zeros((3, 3, 10), dtype=torch.float16),
            "feature_attention_mask": torch.zeros((3, 10), dtype=torch.int64),
        }
        for index, source in enumerate(sources):
            width = source["input_features"].shape[-1]
            expected["input_features"][index, :, :width] = source["input_features"][0]
            expected["feature_attention_mask"][index, :width] = source[
                "feature_attention_mask"
            ][0]

        merged = self.processor.merge_audio_features(sources)

        self.assert_features_equal(merged, expected)
        self.assertEqual(merged["feature_attention_mask"].sum(-1).tolist(), [2, 0, 8])
        merged["input_features"].zero_()
        merged["feature_attention_mask"].zero_()
        for source, before in zip(sources, original):
            self.assert_features_equal(source, before)


if __name__ == "__main__":
    unittest.main()
