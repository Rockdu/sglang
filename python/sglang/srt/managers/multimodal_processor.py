# TODO: also move pad_input_ids into this module
import importlib
import inspect
import logging
import pkgutil

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

PROCESSOR_MAPPING = {}


def _architecture_name(model):
    return model if isinstance(model, str) else model.__name__


def import_processors(package_name: str, overwrite: bool = False):
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            all_members = inspect.getmembers(module, inspect.isclass)
            classes = [
                member
                for name, member in all_members
                if member.__module__ == module.__name__
            ]
            for cls in (
                cls for cls in classes if issubclass(cls, BaseMultimodalProcessor)
            ):
                assert hasattr(cls, "models")
                for arch in getattr(cls, "models"):
                    if overwrite:
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():
                            if _architecture_name(model_cls) == _architecture_name(
                                arch
                            ):
                                del PROCESSOR_MAPPING[model_cls]
                                break
                    PROCESSOR_MAPPING[arch] = cls


def get_mm_processor_cls(hf_config, model_config=None, *, runtime_context=None):
    """The class :func:`get_mm_processor` would instantiate, or ``None`` when the
    architecture has no registered processor."""
    runtime_context = runtime_context or get_context()
    model_impl = str(runtime_context.config_bag("model").model_impl).lower()
    uses_transformers_backend = model_impl == "transformers"
    if model_impl == "auto" and model_config is not None:
        from sglang.srt.model_loader.utils import get_resolved_model_impl

        uses_transformers_backend = (
            get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
        )

    for model_cls, processor_cls in PROCESSOR_MAPPING.items():
        architecture = _architecture_name(model_cls)
        if architecture not in hf_config.architectures:
            continue
        if not uses_transformers_backend or getattr(
            processor_cls, "supports_transformers_backend", False
        ):
            return processor_cls

    if uses_transformers_backend:
        from sglang.srt.multimodal.processors.transformers_auto import (
            TransformersAutoMultimodalProcessor,
        )

        return TransformersAutoMultimodalProcessor

    return None


def get_mm_processor(
    hf_config,
    server_args: ServerArgs,
    processor,
    transport_mode,
    model_config=None,
    *,
    runtime_context=None,
    **kwargs,
) -> BaseMultimodalProcessor:
    processor_cls = get_mm_processor_cls(
        hf_config, model_config, runtime_context=runtime_context
    )
    if processor_cls is None:
        raise ValueError(
            f"No processor registered for architecture: {hf_config.architectures}.\n"
            f"Registered architectures: {[_architecture_name(model_cls) for model_cls in PROCESSOR_MAPPING]}"
        )
    if runtime_context is not None:
        kwargs["runtime_context"] = runtime_context
    return processor_cls(hf_config, server_args, processor, transport_mode, **kwargs)
