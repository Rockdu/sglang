from dataclasses import dataclass


@dataclass
class MultimodalProcessorConfig:
    """Preprocessing and resource settings owned by one processor instance."""

    image_processor_backend: str = "auto"
    disable_fast_image_processor: bool = False
