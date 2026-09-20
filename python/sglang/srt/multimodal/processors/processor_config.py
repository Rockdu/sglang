import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MultimodalProcessorConfig:
    """Preprocessing and resource settings owned by one processor instance."""

    image_processor_backend: str = "auto"
    disable_fast_image_processor: bool = False
    mm_process_config: dict[str, Any] = field(default_factory=dict)
    mm_processor_worker_num: int = 0
    mm_io_worker_num: int = 0
    cpu_worker_num: int = field(
        default_factory=lambda: int(
            os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())
        )
    )
    cpu_process_start_method: str = "spawn"
