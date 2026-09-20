"""Qwen native fields retain source boundaries through splitting and assembly.

    batched native fields                 per-source native fields
    pixels [patches, channels] -- grids -> pixels [source patches, channels]
    grid [sources, 3] -------------------> grid [1, 3]
    video metadata / times -------------> metadata / times [source:source+1]
    audio [sources, bins, padded T] -----> audio [1, bins, same padded T]
    mask [sources, padded T] ------------> mask [1, same padded T]
                                                   |
                                           reorder per-source fields
                                                   |
    native fields <---- concatenate patches/grids; pad audio and masks to max T

Splitting preserves tensor views and the original padded audio width. Merging
retains source order, zero-pads audio masks, and never modifies source inputs.
Splitting and merging one batch must preserve every tensor byte and metadata.
"""

import copy
import unittest

import torch
from transformers.video_utils import VideoMetadata

from sglang.srt.multimodal.modality import Modality
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

    def test_unequal_image_patches_round_trip_without_copying_splits(self):
        features = {
            "pixel_values": torch.arange(84, dtype=torch.float32).reshape(28, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4], [1, 4, 4]]),
        }
        features["pixel_values"][0, 0] = -0.0
        features["pixel_values"][1, 0] = float("nan")
        original = copy.deepcopy(features)

        sources = self.processor.split_media_features(Modality.IMAGE, features)

        self.assertEqual(
            [source["pixel_values"].shape[0] for source in sources], [4, 8, 16]
        )
        for source in sources:
            self.assertEqual(source["image_grid_thw"].shape, (1, 3))
            for name in features:
                self.assertEqual(
                    source[name].untyped_storage().data_ptr(),
                    features[name].untyped_storage().data_ptr(),
                )
        self.assert_features_equal(
            self.processor.merge_media_features(Modality.IMAGE, sources), original
        )
        self.assert_features_equal(features, original)

    def test_video_patches_metadata_and_times_follow_source_order(self):
        for time_name in (None, "second_per_grid_ts", "video_second_per_grid"):
            with self.subTest(time_name=time_name):
                features = {
                    "pixel_values_videos": torch.arange(
                        144, dtype=torch.float32
                    ).reshape(48, 3),
                    "video_grid_thw": torch.tensor([[2, 2, 2], [3, 2, 4], [1, 4, 4]]),
                    "video_metadata": [
                        VideoMetadata(20, fps=8.0, frames_indices=[0, 3, 8, 9]),
                        VideoMetadata(
                            24, fps=12.0, frames_indices=[0, 4, 8, 12, 16, 20]
                        ),
                        VideoMetadata(16, fps=4.0, frames_indices=[0, 8]),
                    ],
                }
                if time_name is not None:
                    features[time_name] = [0.5, 0.25, 1.0]
                original = copy.deepcopy(features)

                sources = self.processor.split_media_features(Modality.VIDEO, features)

                self.assertEqual(
                    [source["pixel_values_videos"].shape[0] for source in sources],
                    [8, 24, 16],
                )
                for index, source in enumerate(sources):
                    self.assertEqual(source["video_grid_thw"].shape, (1, 3))
                    self.assertIs(
                        source["video_metadata"][0], features["video_metadata"][index]
                    )
                self.assert_features_equal(
                    self.processor.merge_media_features(Modality.VIDEO, sources),
                    original,
                )
                ordered = self.processor.merge_media_features(
                    Modality.VIDEO, [sources[2], sources[0], sources[1]]
                )
                expected = {
                    "pixel_values_videos": torch.cat(
                        [
                            original["pixel_values_videos"][32:],
                            original["pixel_values_videos"][:8],
                            original["pixel_values_videos"][8:32],
                        ]
                    ),
                    "video_grid_thw": original["video_grid_thw"][[2, 0, 1]],
                    "video_metadata": [
                        original["video_metadata"][index] for index in (2, 0, 1)
                    ],
                }
                if time_name is not None:
                    expected[time_name] = [1.0, 0.5, 0.25]
                self.assert_features_equal(ordered, expected)
                self.assert_features_equal(features, original)

    def test_audio_round_trip_preserves_padded_values_and_mask_counts(self):
        features = {
            "input_features": torch.arange(90, dtype=torch.float16).reshape(3, 3, 10),
            "feature_attention_mask": torch.tensor(
                [[1] * 2 + [0] * 8, [1] * 7 + [0] * 3, [0] * 10]
            ),
        }
        original = copy.deepcopy(features)

        sources = self.processor.split_media_features(Modality.AUDIO, features)

        for source in sources:
            self.assertEqual(source["input_features"].shape, (1, 3, 10))
            self.assertEqual(source["feature_attention_mask"].shape, (1, 10))
            for name in features:
                self.assertEqual(
                    source[name].untyped_storage().data_ptr(),
                    features[name].untyped_storage().data_ptr(),
                )
        merged = self.processor.merge_media_features(Modality.AUDIO, sources)
        self.assert_features_equal(merged, original)
        self.assertEqual(merged["feature_attention_mask"].sum(-1).tolist(), [2, 7, 0])
        self.assert_features_equal(features, original)

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

        merged = self.processor.merge_media_features(Modality.AUDIO, sources)

        self.assert_features_equal(merged, expected)
        self.assertEqual(merged["feature_attention_mask"].sum(-1).tolist(), [2, 0, 8])
        merged["input_features"].zero_()
        merged["feature_attention_mask"].zero_()
        for source, before in zip(sources, original):
            self.assert_features_equal(source, before)


if __name__ == "__main__":
    unittest.main()
