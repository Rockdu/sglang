"""Media splitting preserves source boundaries and tensor views."""

# Packed feature: [source A patches | source B patches]
#                  + grids + offsets=None -> source views, offsets remain None
#                  + image offsets        -> one offset per source
#                  + video frame offsets  -> all frame offsets per source
# Missing/malformed grids cannot establish source boundaries without offsets.

import unittest

import numpy as np
import torch

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _bundled_item(grid_key=None, grid=None, feature_len=10, num_images=2):
    """A bundled IMAGE item: `num_images` offsets, one concatenated feature."""
    model_specific_data = {}
    if grid_key is not None:
        model_specific_data[grid_key] = grid
    # Distinct per-row values so slice boundaries are checkable.
    feature = torch.arange(feature_len * 3, dtype=torch.float32).reshape(feature_len, 3)
    offsets = [(0, 5), (5, feature_len)][:num_images]
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=feature,
        model_specific_data=model_specific_data,
    )


class TestGetNewExpandedMMItems(CustomTestCase):
    def test_grids_split_source_views_without_offsets(self):
        for modality, grid_key, grid in (
            (Modality.IMAGE, "image_grid_thw", [[1, 2, 3], [1, 4, 1]]),
            (Modality.IMAGE, "image_grid_hws", [[2, 3], [4, 1]]),
            (Modality.VIDEO, "video_grid_thw", [[2, 1, 3], [1, 2, 2]]),
        ):
            with self.subTest(grid_key=grid_key):
                feature = torch.arange(30, dtype=torch.float32).reshape(10, 3)
                grids = torch.tensor(grid)
                patch_mask = torch.arange(10)
                bundled = MultimodalDataItem(
                    modality=modality,
                    feature=feature,
                    hash=123,
                    model_specific_data={grid_key: grids, "patch_mask": patch_mask},
                )
                sources = get_new_expanded_mm_items([bundled])

                self.assertEqual(len(sources), 2)
                for index, (source, start, end) in enumerate(
                    zip(sources, (0, 6), (6, 10))
                ):
                    self.assertIsNone(source.offsets)
                    self.assertIsNone(source.hash)
                    for view, tensor, expected in (
                        (source.feature, feature, feature[start:end]),
                        (
                            source.model_specific_data[grid_key],
                            grids,
                            grids[index : index + 1],
                        ),
                        (
                            source.model_specific_data["patch_mask"],
                            patch_mask,
                            patch_mask[start:end],
                        ),
                    ):
                        self.assertTrue(torch.equal(view, expected))
                        self.assertEqual(
                            view.untyped_storage().data_ptr(),
                            tensor.untyped_storage().data_ptr(),
                        )
                self.assertIsNone(bundled.offsets)
                self.assertEqual(bundled.hash, 123)

    def test_offsetless_features_do_not_establish_source_boundaries(self):
        for modality, model_specific_data in (
            (Modality.IMAGE, {}),
            (Modality.AUDIO, {}),
            (Modality.IMAGE, {"image_grid_thw": torch.tensor([2, 2])}),
            (Modality.VIDEO, {"video_grid_thw": torch.tensor([2, 2, 2])}),
            (
                Modality.IMAGE,
                {"image_grid_thw": torch.tensor([[1, 2, 3], [1, 4, 1]])},
            ),
        ):
            with self.subTest(
                modality=modality, model_specific_data=model_specific_data
            ):
                bundled = MultimodalDataItem(
                    modality=modality,
                    feature=torch.ones(2, 3),
                    model_specific_data=model_specific_data,
                )
                sources = get_new_expanded_mm_items([bundled])
                self.assertEqual(len(sources), 1)
                self.assertIs(sources[0], bundled)

    def test_bound_video_offsets_keep_legacy_frame_grouping(self):
        feature = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        frame_offsets = [(1, 3), (5, 7), (9, 12)]
        bundled = MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=feature,
            offsets=frame_offsets,
            model_specific_data={
                "video_grid_thw": torch.tensor([[2, 1, 3], [1, 2, 2]])
            },
        )
        sources = get_new_expanded_mm_items([bundled])
        self.assertEqual(
            [source.offsets for source in sources],
            [frame_offsets[:2], frame_offsets[2:]],
        )
        self.assertTrue(torch.equal(sources[0].feature, feature[:6]))
        self.assertTrue(torch.equal(sources[1].feature, feature[6:]))

        bundled.offsets = [(1, 6), (8, 11)]
        sources = get_new_expanded_mm_items([bundled])
        self.assertEqual(len(sources), 1)
        self.assertIs(sources[0], bundled)

    def test_image_grid_hws_splits_per_image(self):
        # grid rows [[2,3],[4,1]] -> prod = [6, 4] patches -> feature_len 10.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=[[2, 3], [4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertEqual([len(o.offsets) for o in out], [1, 1])
        self.assertEqual(out[0].offsets, [(0, 5)])
        self.assertEqual(out[1].offsets, [(5, 10)])
        # Feature sliced 0:6 and 6:10 along dim-0.
        self.assertEqual(out[0].feature.shape[0], 6)
        self.assertEqual(out[1].feature.shape[0], 4)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))
        # Split items must re-hash (pad value is recomputed per image).
        self.assertTrue(all(o.hash is None for o in out))

    def test_image_grid_hws_tensor_splits_per_image(self):
        # Same as above but the grid arrives as a rank-2 tensor (HF emits these).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([[2, 3], [4, 1]], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_image_grid_thw_still_splits(self):
        # The pre-existing image_grid_thw path must keep working:
        # [[1,2,3],[1,4,1]] -> [6,4].
        item = _bundled_item(
            grid_key="image_grid_thw",
            grid=[[1, 2, 3], [1, 4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_missing_grid_falls_back_to_simple_split(self):
        # No grid, but feature dim-0 == num offsets -> simple per-row split.
        item = _bundled_item(grid_key=None, feature_len=2, num_images=2)
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:1]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[1:2]))

    def test_flat_1d_grid_does_not_mis_split(self):
        # A flat 1-D grid (`tensor([2, 2])`) has length == num_items so it passes
        # the length check, but prod(dim=-1) would collapse it to a scalar and
        # corrupt the slice boundaries. The rank-2 guard must reject it. With
        # feature_len != num_items, the simple-split fallback also declines, so
        # the bundled item is passed through unchanged (never mis-sliced).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([2, 2], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)

    def test_numpy_grid_splits_per_image(self):
        # image_grid_hws can arrive as a numpy array from the HF image processor.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=np.array([[2, 3], [4, 1]], dtype=np.int64),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_non_bundled_item_passes_through(self):
        # A single-image item (one offset) is not bundled and is returned as-is.
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 5)],
            feature=torch.arange(18, dtype=torch.float32).reshape(6, 3),
            model_specific_data={"image_grid_hws": [[2, 3]]},
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)


if __name__ == "__main__":
    unittest.main()
