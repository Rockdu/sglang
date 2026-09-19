# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, TypeAlias

import msgspec
import torch

from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    CudaIpcTensorTransportProxy,
)
from sglang.srt.utils import flatten_nested_list

# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
MM_PAD_SHIFT_VALUE = 1_000_000
_MM_HASH_MASK = (1 << 64) - 1

logger = logging.getLogger(__name__)


def _compute_pad_value(hash: int) -> int:
    """Compute pad value from hash."""
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


class MultimodalInputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


# Msgpack-native containers and Ext-decoded tensor/transport leaves. Tuple
# containers intentionally decode as lists, matching msgpack's native model.
MultimodalDataValue: TypeAlias = object


class MultimodalDataItem(msgspec.Struct, kw_only=True, dict=True, array_like=True):
    """
    One MultimodalDataItem represents a single multimodal input (one image, one video, or one audio).
    For example, if there are 3 images and 1 audio, there will be 4 MultimodalDataItems.

    Each item has its own hash and pad_value, enabling per-image RadixAttention caching.

    We put the common fields first and the model-specific fields in model_specific_data.
    """

    modality: Modality
    hash: Optional[int] = None
    pad_value: Optional[int] = None
    offsets: Optional[List[Tuple[int, int]]] = None

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

    # the raw features returned by processor, e.g. pixel_values or audio_features
    feature: Optional[MultimodalDataValue] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[MultimodalDataValue] = None
    # Keep precomputed_embeddings on GPU after use (EPD pool/GPU receive path)
    keep_device_embedding: bool = False

    # Processor-owned tensors/arrays/scalars/transports. msgspec rejects a
    # precise union with multiple custom types, but accepts Ext-decoded values
    # under object.
    model_specific_data: Dict[str, MultimodalDataValue] = msgspec.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.hash is not None:
            msgspec.Struct.__setattr__(self, "hash", self.hash & _MM_HASH_MASK)

    def __getattr__(self, name: str) -> MultimodalDataValue:
        if name in self.model_specific_data:
            return self.model_specific_data[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: MultimodalDataValue) -> None:
        if name in self.__struct_fields__:
            if name == "hash" and isinstance(value, int):
                value &= _MM_HASH_MASK
            msgspec.Struct.__setattr__(self, name, value)
        else:
            self.model_specific_data[name] = value

    def __setitem__(self, key: str, value: MultimodalDataValue) -> None:
        setattr(self, key, value)

    def set(self, key: str, value: MultimodalDataValue) -> None:
        self.__setitem__(key, value)

    def set_hash(self, hash_value: int) -> None:
        self.hash = hash_value
        self.pad_value = _compute_pad_value(hash_value)

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        if self.pad_value is not None:
            return

        from sglang.srt.multimodal.cache import resolve_multimodal_item_hash

        self.hash = resolve_multimodal_item_hash(
            existing_hash=self.hash,
            feature=self.feature,
            precomputed_embeddings=self.precomputed_embeddings,
        )
        self.pad_value = _compute_pad_value(self.hash)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    def is_precomputed_embedding(self):
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def has_cuda_ipc_proxy(self):
        return (
            isinstance(self.feature, CudaIpcTensorTransportProxy)
            or isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            or any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def reconstruct(self, target_device: int, ipc_consumer_count: int = 1):
        """materialize cuda ipc proxy tensors in-place on target_device"""
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            consumer_count = self._resolve_transport_consumer_count(
                self.feature, ipc_consumer_count
            )
            if consumer_count == 1:
                self.feature = self.feature.reconstruct_on_target_device(target_device)
            else:
                self.feature = self.feature.reconstruct_on_target_device(
                    target_device, consumer_count=consumer_count
                )
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(target_device)
            )
        for extra_key in self.model_specific_data:
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(target_device)
                self.model_specific_data[extra_key] = extra_data

    def can_defer_cuda_ipc_feature_reconstruction(self) -> bool:
        """Whether a DP-aware model will materialize this feature lazily.

        Hashing and pad-value generation must already have completed on the
        tokenizer worker.  Any additional IPC proxy would still need eager
        reconstruction, so keep the narrow fast path feature-only.
        """
        return (
            self.model_specific_data.get(
                DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY, False
            )
            and self.hash is not None
            and self.pad_value is not None
            and isinstance(self.feature, CudaIpcTensorTransportProxy)
            and not isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy)
            and not any(
                isinstance(value, CudaIpcTensorTransportProxy)
                for value in self.model_specific_data.values()
            )
        )

    def acknowledge_deferred_cuda_ipc_feature(self, consumer_count: int = 1):
        """Release a lazy IPC feature when an embedding-cache hit skips ViT."""
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            consumer_count = self._resolve_transport_consumer_count(
                self.feature, consumer_count
            )
            self.feature.acknowledge_consumption(consumer_count)

    def release_transport_proxies(self, consumer_count: int = 1) -> None:
        """Best-effort release of proxies left by an abandoned request."""
        values = [self.feature, self.precomputed_embeddings]
        values.extend(self.model_specific_data.values())
        for value in values:
            if not isinstance(value, CudaIpcTensorTransportProxy):
                continue
            count = self._resolve_transport_consumer_count(value, consumer_count)
            try:
                value.release_without_reconstruction(count)
            except Exception:
                logger.warning(
                    "Failed to release an abandoned multimodal transport proxy",
                    exc_info=True,
                )

    @staticmethod
    def _resolve_transport_consumer_count(proxy, requested_count: int) -> int:
        """Clamp a group acknowledgement to the proxy's actual consumer set."""
        proxy_count = getattr(
            proxy,
            "total_consumer_count",
            getattr(proxy, "consumer_count", requested_count),
        )
        return min(requested_count, proxy_count)


class MultimodalProcessorOutput(
    msgspec.Struct, kw_only=True, dict=True, array_like=True, weakref=True
):
    """Raw output from multimodal processors before scheduler-side preparation (pad, hash).

    This is the typed replacement for the dict previously returned by
    ``BaseMultimodalProcessor.process_mm_data_async``.  Preprocessed inputs may
    already carry ``pad_value`` and ``hash`` to avoid hashing the same tensor once
    per scheduler TP rank.
    """

    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None
    padded_input_ids: Optional[List[int]] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    # for transformers-compatibility
    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: dict) -> MultimodalProcessorOutput:
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            padded_input_ids=d.get("padded_input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
            vision_position_ids=d.get("vision_position_ids"),
            media_nums_per_sample=d.get("media_nums_per_sample"),
            visible_frame_counts=d.get("visible_frame_counts"),
        )

    @staticmethod
    def build_padded_input_ids(input_ids, mm_items: List[MultimodalDataItem]):
        """pad the input_ids with mm_items if it's not already padded"""
        if input_ids is None or not mm_items:
            return None

        for item in mm_items:
            if item.pad_value is None or item.offsets is None:
                return None

        if isinstance(input_ids, torch.Tensor):
            padded_input_ids = input_ids.flatten().tolist()
        else:
            padded_input_ids = list(input_ids)

        for item in mm_items:
            for start, end in item.offsets:
                padded_input_ids[start : end + 1] = [item.pad_value] * (end - start + 1)
        return padded_input_ids
