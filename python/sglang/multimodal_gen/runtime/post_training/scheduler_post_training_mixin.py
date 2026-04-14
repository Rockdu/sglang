from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler


class SchedulerPostTrainingMixin:
    def _handle_update_weights_from_disk(
        self: Scheduler, reqs: List[Any]
    ) -> OutputBatch:
        if self.worker.is_sleeping():
            raise RuntimeError(
                "Cannot update weights while the server is sleeping. "
                "Call resume_memory_occupation first."
            )
        req = reqs[0]
        success, message = self.worker.update_weights_from_disk(
            model_path=req.model_path,
            flush_cache=req.flush_cache,
            target_modules=req.target_modules,
        )
        return OutputBatch(
            output={"success": success, "message": message},
            error=None if success else message,
        )
