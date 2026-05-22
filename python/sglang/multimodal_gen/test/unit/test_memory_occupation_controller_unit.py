"""CPU-only unit tests for MemoryOccupationController.

Guards `_offload_active_modules_to_cpu` against regressions in the
layerwise-skip branch: modules whose `is_layerwise_offloaded_module(...)`
predicate returns True must not be queued for the sleep offload, because
they are already governed by a `LayerwiseOffloadManager`. The test uses
hand-built fake modules so it does not need a GPU.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.managers.memory_occupation_controller import (
    MemoryOccupationController,
)


class _FakeEnabledManager:
    """Stand-in for `LayerwiseOffloadManager` exposing only the bits the
    predicate `is_layerwise_offloaded_module` reads."""

    enabled = True


class _FakeLayerwiseManagedModule(LayerwiseOffloadableModuleMixin, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.layerwise_offload_managers = [_FakeEnabledManager()]


class _PlainModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class TestMemoryOccupationControllerLayerwiseSkip(unittest.TestCase):
    """release_memory_occupation must skip modules already governed by
    layerwise offload (i.e., `is_layerwise_offloaded_module(module) is True`)
    and only enqueue plain modules for CPU offload."""

    def _make_controller(self):
        pipeline = object()
        return MemoryOccupationController(
            pipeline=pipeline,
            rank=0,
            use_fsdp_inference=False,
        )

    def test_layerwise_managed_module_skipped_release(self):
        controller = self._make_controller()
        plain = _PlainModule()
        layerwise = _FakeLayerwiseManagedModule()
        fake_modules = {
            "plain": plain,
            "layerwise": layerwise,
        }

        with patch(
            "sglang.multimodal_gen.runtime.managers.memory_occupation_controller.get_updatable_modules",
            return_value=fake_modules,
        ), patch(
            "sglang.multimodal_gen.runtime.managers.memory_occupation_controller._get_module_device",
            return_value="cuda:0",
        ), patch.object(
            MemoryOccupationController, "_move_modules", autospec=True
        ) as move_mock, patch.object(
            MemoryOccupationController, "_clear_torch_device_cache", autospec=True
        ):
            result = controller.release_memory_occupation()

        self.assertTrue(result["success"])
        self.assertTrue(result["sleeping"])
        self.assertEqual(controller._sleep_restore_map, {"plain": "cuda:0"})
        move_mock.assert_called_once()
        moved_names = move_mock.call_args.args[1]
        self.assertEqual(moved_names, ["plain"])

    def test_layerwise_managed_module_skipped_predicate_check(self):
        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (  # noqa: I001
            is_layerwise_offloaded_module,
        )

        self.assertTrue(is_layerwise_offloaded_module(_FakeLayerwiseManagedModule()))
        self.assertFalse(is_layerwise_offloaded_module(_PlainModule()))


if __name__ == "__main__":
    unittest.main()
