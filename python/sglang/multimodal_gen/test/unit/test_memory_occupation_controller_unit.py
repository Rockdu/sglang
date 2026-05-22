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


class _ToSpyModule(torch.nn.Module):
    """Base class that records every `to(device)` invocation in `to_calls`.
    Used by both fake module types so the test can verify the layerwise-managed
    module's device is never touched and the plain module is moved to 'cpu'.

    `to()` is overridden to ONLY record the requested device and return self,
    without delegating to `nn.Module.to`. This keeps the test CPU-safe: a real
    `super().to('cuda:0')` from `resume_memory_occupation` would otherwise fail
    on a CPU-only runner where CUDA is not available. The spy is sufficient
    because the assertions are about the call sequence, not the tensor device
    state after the move."""

    def __init__(self):
        super().__init__()
        self.to_calls: list[str] = []

    def to(self, device, *args, **kwargs):  # type: ignore[override]
        self.to_calls.append(str(device))
        return self


class _FakeLayerwiseManagedModule(LayerwiseOffloadableModuleMixin, _ToSpyModule):
    def __init__(self):
        _ToSpyModule.__init__(self)
        self.layerwise_offload_managers = [_FakeEnabledManager()]


class _PlainModule(_ToSpyModule):
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

    def test_release_moves_plain_module_to_cpu_and_skips_layerwise(self):
        """Behavior assertion: after release_memory_occupation,
        - the plain module's recorded `to()` invocation is to 'cpu', AND
        - the layerwise-managed module's `to()` is never invoked.

        Also asserts the controller's restore map remembers the plain module's
        source device ('cuda:0', supplied via the patched _get_module_device)
        so a subsequent resume_memory_occupation will return it to the right
        device."""
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
            MemoryOccupationController, "_clear_torch_device_cache", autospec=True
        ):
            # _move_modules NOT patched — real implementation runs and calls
            # plain.to("cpu") while leaving the layerwise module untouched.
            result = controller.release_memory_occupation()

        self.assertTrue(result["success"])
        self.assertTrue(result["sleeping"])
        self.assertEqual(controller._sleep_restore_map, {"plain": "cuda:0"})
        # Behavior: plain module was moved to CPU exactly once
        self.assertEqual(plain.to_calls, ["cpu"])
        # Behavior: layerwise-managed module was never moved
        self.assertEqual(layerwise.to_calls, [])

    def test_resume_restores_plain_module_to_source_device(self):
        """Round-trip behavior: a release_memory_occupation followed by a
        resume_memory_occupation must invoke .to('cuda:0') on the plain
        module (its recorded source device) and still leave the
        layerwise-managed module untouched."""
        controller = self._make_controller()
        plain = _PlainModule()
        layerwise = _FakeLayerwiseManagedModule()
        fake_modules = {
            "plain": plain,
            "layerwise": layerwise,
        }

        # Pre-load the restore map as if a release already happened, then
        # patch _move_modules to a real call-through for the resume.
        with patch(
            "sglang.multimodal_gen.runtime.managers.memory_occupation_controller.get_updatable_modules",
            return_value=fake_modules,
        ), patch(
            "sglang.multimodal_gen.runtime.managers.memory_occupation_controller._get_module_device",
            return_value="cuda:0",
        ), patch.object(
            MemoryOccupationController, "_clear_torch_device_cache", autospec=True
        ):
            controller.release_memory_occupation()
            # Clear the recorded sleep-time .to() so the assertion below is
            # unambiguous about the resume-path call.
            plain.to_calls.clear()
            result = controller.resume_memory_occupation()

        self.assertTrue(result["success"])
        self.assertFalse(result["sleeping"])
        self.assertEqual(plain.to_calls, ["cuda:0"])
        self.assertEqual(layerwise.to_calls, [])

    def test_predicate_directly(self):
        """Direct predicate sanity check: the upstream
        is_layerwise_offloaded_module helper must return True for our fake
        managed module and False for a plain torch.nn.Module."""
        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (  # noqa: I001
            is_layerwise_offloaded_module,
        )

        self.assertTrue(is_layerwise_offloaded_module(_FakeLayerwiseManagedModule()))
        self.assertFalse(is_layerwise_offloaded_module(_PlainModule()))


if __name__ == "__main__":
    unittest.main()
