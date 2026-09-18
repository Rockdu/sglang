"""Pure CPU contracts for independently reusable media pipeline stages.

  input IDs: [expanded history | placeholders]    full-dialogue media fragments
                              A                  images / video frames / audio
                              |                              |
                         suffix-only expansion <-------------+
                              |
                 final IDs + new-media bindings
                              |
                     independent full build
                              |
                 complete per-media token ranges

Checks cover multi-token patterns, rule priority, adjacent images with identical
pads, video nesting and timestamps, historical layout changes, no-op suffixes,
unmodified replacement rules, empty media, and per-source grid views.
Processor metadata stays outside the per-source encoder fields sent to serving.
These helpers import no serving scheduler or model classes.
"""

import copy
import unittest

import torch

from sglang.srt.multimodal.media_processing import (
    collect_media_bindings,
    pack_grid_media_output,
)
from sglang.srt.multimodal.mm_token_expansion import (
    expand_mm_tokens,
    expand_token_placeholders,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMediaProcessing(unittest.TestCase):
    def test_empty_encoder_fragment_preserves_empty_media_binding(self):
        replacements = [([7], [[([], ("audio", 0))]])]
        expanded = expand_mm_tokens([10, 7, 11], replacements)
        self.assertEqual(expanded.input_ids, [10, 11])
        self.assertEqual(
            collect_media_bindings(expanded.input_ids, replacements), {("audio", 0): []}
        )

    def test_mixed_media_and_adjacent_images_keep_source_identity(self):
        # [I I V A] -> [ii][iii][<frame ii> t <frame ii>][aaaa]
        # Video uses the image pad, but whole-video matching owns both frames.
        image0, image1 = ("image", 0), ("image", 1)
        video, audio = ("video", 0), ("audio", 0)
        replacements = [
            ([9], [[([9, 9], image0)], [([9, 9, 9], image1)]]),
            (
                [20, 8, 21],
                [
                    [
                        ([20], None),
                        ([9, 9], video),
                        ([21, 40, 20], None),
                        ([9, 9], video),
                        ([21], None),
                    ]
                ],
            ),
            ([7], [[([7] * 4, audio)]]),
        ]
        original_ids = [9, 9, 20, 8, 21, 7]
        original_replacements = copy.deepcopy(replacements)
        expanded = expand_mm_tokens(original_ids, replacements)
        expected_bindings = {
            image0: [(0, 2)],
            image1: [(2, 5)],
            video: [(6, 8), (11, 13)],
            audio: [(14, 18)],
        }
        self.assertEqual(
            expanded.input_ids,
            [9, 9, 9, 9, 9, 20, 9, 9, 21, 40, 20, 9, 9, 21, 7, 7, 7, 7],
        )
        self.assertEqual(expanded.new_media_bindings, expected_bindings)
        self.assertEqual(
            collect_media_bindings(
                expanded.input_ids, replacements, expanded.new_media_bindings
            ),
            expected_bindings,
        )
        self.assertEqual(original_ids, [9, 9, 20, 8, 21, 7])
        self.assertEqual(replacements, original_replacements)

    def test_partial_expansion_only_uses_trailing_media(self):
        image0, image1, audio = ("image", 0), ("image", 1), ("audio", 0)
        history = [10, 9, 9, 11, 7, 7, 12]
        replacements = [
            ([9], [[([9, 9], image0)], [([9, 9, 9], image1)]]),
            ([7], [[([7, 7], audio)]]),
        ]
        partial = expand_mm_tokens(history + [9], replacements, len(history))
        full = expand_mm_tokens([10, 9, 11, 7, 12, 9], replacements)
        self.assertEqual(partial.input_ids, full.input_ids)
        self.assertEqual(partial.input_ids[: len(history)], history)
        self.assertEqual(partial.new_media_bindings, {image1: [(7, 10)]})
        self.assertEqual(
            collect_media_bindings(
                partial.input_ids, replacements, partial.new_media_bindings
            ),
            full.new_media_bindings,
        )
        # A suffix with no image or audio must not select all media via [-0:].
        no_new_media = expand_mm_tokens(
            full.input_ids, replacements, len(full.input_ids)
        )
        self.assertEqual(no_new_media.input_ids, full.input_ids)
        self.assertEqual(no_new_media.new_media_bindings, {})
        self.assertEqual(
            collect_media_bindings(no_new_media.input_ids, replacements),
            full.new_media_bindings,
        )

    def test_expansion_does_not_parse_or_validate_history(self):
        # Deliberately invalid history is untouched by expansion; build rejects it.
        original = [9, 9, 9, 10, 9]
        replacements = [([9], [[([9, 9], ("image", 0))], [([9], ("image", 1))]])]
        expanded = expand_mm_tokens(original, replacements, 4)
        self.assertEqual(expanded.input_ids, original)
        self.assertEqual(expanded.new_media_bindings, {("image", 1): [(4, 5)]})
        with self.assertRaisesRegex(ValueError, "historical media"):
            collect_media_bindings(
                expanded.input_ids, replacements, expanded.new_media_bindings
            )

    def test_generated_outer_fragment_takes_precedence_over_image_pads(self):
        # Even image-first rules must not claim a video's nested image frame.
        video = ("video", 0)
        replacements = [
            ([9], [[([9, 9], ("image", 0))]]),
            ([8], [[([9, 9], video), ([40], None), ([9, 9], video)]]),
        ]
        expanded = expand_mm_tokens([8, 10, 9], replacements)
        self.assertEqual(
            collect_media_bindings(expanded.input_ids, replacements),
            {video: [(0, 2), (3, 5)], ("image", 0): [(6, 8)]},
        )
        ambiguous = [
            ([9], [[([9, 9], ("image", 0))]]),
            ([8], [[([9, 9], video)]]),
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            collect_media_bindings([9, 9, 9, 9], ambiguous)

    def test_build_rejects_missing_changed_and_misbound_fragments(self):
        replacements = [
            ([8], [[([20], None), ([9, 9], ("video", 0)), ([40, 21], None)]])
        ]
        for invalid_ids in ([20, 9, 9, 41, 21], [20, 9, 21], []):
            with (
                self.subTest(ids=invalid_ids),
                self.assertRaisesRegex(ValueError, "historical media"),
            ):
                collect_media_bindings(invalid_ids, replacements)
        with self.assertRaisesRegex(ValueError, "New media bindings"):
            collect_media_bindings(
                [20, 9, 9, 40, 21], replacements, {("video", 0): [(2, 4)]}
            )

    def test_id_only_api_uses_same_rule_priority_and_never_rescans_insertions(self):
        original = [80, 9, 9, 80]
        id_replacements = [([80, 9], [[7, 9]]), ([9], [[9, 9]]), ([80], [[80, 9]])]
        annotated = [
            (pattern, [[(fragment, None)] for fragment in fragments])
            for pattern, fragments in id_replacements
        ]
        self.assertEqual(
            expand_token_placeholders(original, id_replacements), [7, 9, 9, 9, 80, 9]
        )
        self.assertEqual(
            expand_token_placeholders(original, id_replacements),
            expand_mm_tokens(original, annotated).input_ids,
        )
        for boundary in (-1, len(original) + 1, 1.5):
            with (
                self.subTest(boundary=boundary),
                self.assertRaisesRegex(ValueError, "mm_token_expansion_start_len"),
            ):
                expand_mm_tokens(original, annotated, boundary)
        for ids, rules in [([], annotated), ([9], []), ([9], [([], [])])]:
            if not rules:
                self.assertEqual(expand_mm_tokens(ids, rules).input_ids, ids)
            else:
                with self.assertRaises(ValueError):
                    expand_mm_tokens(ids, rules)

    def test_grid_packing_keeps_tensor_views_and_explicit_metadata(self):
        # Batched patches [image0:2 | image1:3] retain a single backing tensor.
        pixels = torch.arange(10).reshape(5, 2)
        inputs = {
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
            "sampling_times": [[0.0], [1.0]],
        }
        output = pack_grid_media_output(
            "image",
            inputs,
            "pixel_values",
            "image_grid_thw",
            metadata_names=("sampling_times",),
        )
        self.assertIs(output.encoder_inputs, inputs)
        self.assertEqual(
            [item.media_id for item in output.items], [("image", 0), ("image", 1)]
        )
        self.assertTrue(
            torch.equal(output.items[1].encoder_inputs["pixel_values"], pixels[2:])
        )
        self.assertEqual(
            output.items[1].encoder_inputs["pixel_values"].untyped_storage().data_ptr(),
            pixels.untyped_storage().data_ptr(),
        )
        self.assertIs(
            output.items[1].metadata["sampling_times"], inputs["sampling_times"][1]
        )
        self.assertNotIn("sampling_times", output.items[1].encoder_inputs)


if __name__ == "__main__":
    unittest.main()
