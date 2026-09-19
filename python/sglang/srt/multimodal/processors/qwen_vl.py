import math
import os
import re
import time
from dataclasses import asdict, replace
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode
from transformers import BaseImageProcessor
from transformers.image_utils import SizeDict
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessorKwargs,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as smart_resize_video,
)

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.cosmos3 import Cosmos3ForConditionalGeneration
from sglang.srt.models.interns2_mobius import (
    InternS2MobiusForConditionalGeneration,
)
from sglang.srt.models.interns2preview import InternS2PreviewForConditionalGeneration
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.models.qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    _get_feat_extract_output_lengths,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from sglang.srt.multimodal.media_processing import (
    MediaProcessOutput,
    ProcessedMediaItem,
    process_media_groups,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)
from sglang.srt.utils import cpu_has_amx_support, is_cpu
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.utils import logger

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:
    logger.warning(
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "
        f"Ignoring and using default."
    )
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_VIDEO_DECODE_CHUNK_BYTES = 512 * 1024 * 1024

QWEN_VIDEO_PREPROCESS_CONFIG_KEYS = frozenset(
    {
        "fps",
        "nframes",
        "min_frames",
        "max_frames",
        "min_pixels",
        "max_pixels",
        "total_pixels",
        "resized_height",
        "resized_width",
    }
)


def _get_processor_video_config(video_config, video_metadata):
    if video_metadata and all(metadata is not None for metadata in video_metadata):
        return {
            key: value
            for key, value in video_config.items()
            if key not in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
        }
    return None


_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
if _is_cpu and _is_cpu_amx_available:
    try:
        import transformers

        from sglang.srt.layers.amx_utils import fast_preprocess_cpu

        transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast._preprocess = fast_preprocess_cpu
    except Exception as e:
        logger.warning(
            f"Failed to hack Qwen2VLImageProcessorFast with AMX optimization: {e}"
        )


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), (
        "Only accept either `fps` or `nframes`"
    )
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


def _resize_native_video(
    video, num_frames, video_processor, video_config, processor_kwargs
):
    device = processor_kwargs.get("device")
    if device is not None:
        video = video.to(device)
    if processor_kwargs.get("do_convert_rgb", video_processor.do_convert_rgb):
        video = video_processor.convert_to_rgb(video)
    do_resize = video_config.get(
        "do_resize", processor_kwargs.get("do_resize", video_processor.do_resize)
    )
    if do_resize:
        if "resized_height" in video_config and "resized_width" in video_config:
            resized_height = video_config["resized_height"]
            resized_width = video_config["resized_width"]
        else:
            height, width = video.shape[-2:]
            size = processor_kwargs.get("size", video_processor.size)
            resized_height, resized_width = smart_resize_video(
                num_frames=num_frames,
                height=height,
                width=width,
                temporal_factor=processor_kwargs.get(
                    "temporal_patch_size", video_processor.temporal_patch_size
                ),
                factor=processor_kwargs.get("patch_size", video_processor.patch_size)
                * processor_kwargs.get("merge_size", video_processor.merge_size),
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
            )
        video = video_processor.resize(
            video,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=processor_kwargs.get("resample", video_processor.resample),
        )
    return video


def _decode_native_video(vr, indices, video_processor, video_config, processor_kwargs):
    height, width = vr.frame_shape
    frames_per_chunk = max(1, MAX_VIDEO_DECODE_CHUNK_BYTES // (height * width * 3))
    video = None
    for start in range(0, len(indices), frames_per_chunk):
        chunk_indices = indices[start : start + frames_per_chunk].tolist()
        chunk = vr.get_frames_as_tensor(chunk_indices).permute(0, 3, 1, 2).contiguous()
        chunk = _resize_native_video(
            chunk, len(indices), video_processor, video_config, processor_kwargs
        )
        if video is None:
            video = chunk.new_empty((len(indices), *chunk.shape[1:]))
        video[start : start + len(chunk_indices)].copy_(chunk)
    return video


def preprocess_video_sync(
    vr,
    *,
    image_factor=IMAGE_FACTOR,
    video_config=None,
    video_processor=None,
    processor_kwargs=None,
    resize_raw_frames=False,
):
    video_config = video_config or {}
    processor_kwargs = processor_kwargs or {}
    if isinstance(video_processor, Qwen3VLVideoProcessor):
        legacy_size_keys = {
            "min_pixels",
            "max_pixels",
            "total_pixels",
        } & video_config.keys()
        if legacy_size_keys:
            raise ValueError(
                f"Qwen3 video sizing does not support {sorted(legacy_size_keys)}; "
                "use size with shortest_edge and longest_edge."
            )
    if not isinstance(vr, VideoDecoderWrapper):
        if resize_raw_frames and isinstance(video_processor, Qwen3VLVideoProcessor):
            videos, metadata = video_processor._decode_and_sample_videos(
                vr, video_metadata=None, do_sample_frames=False
            )
            video = video_processor._prepare_input_videos(
                videos,
                input_data_format=processor_kwargs.get("input_data_format"),
                device=processor_kwargs.get("device"),
            )[0]
            return (
                _resize_native_video(
                    video, len(video), video_processor, video_config, processor_kwargs
                ),
                asdict(metadata[0]),
            )
        return vr, None
    total_frames, video_fps = len(vr), vr.avg_fps
    if "frame_indices" in video_config:
        indices = np.asarray(video_config["frame_indices"], dtype=np.int64)
    else:
        nframes = smart_nframes(
            video_config, total_frames=total_frames, video_fps=video_fps
        )
        indices = np.unique(
            np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
        )
    metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": indices,
        "video_backend": "torchvision",
    }
    if isinstance(video_processor, Qwen3VLVideoProcessor):
        return (
            _decode_native_video(
                vr, indices, video_processor, video_config, processor_kwargs
            ),
            metadata,
        )
    video = vr.get_frames_as_tensor(indices.tolist()).permute(0, 3, 1, 2)
    nframes, _, height, width = video.shape
    min_pixels = video_config.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = video_config.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(
            video_config.get("max_pixels", VIDEO_MAX_PIXELS),
            total_pixels / nframes * FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )

    max_pixels_supposed = video_config.get("max_pixels", max_pixels)

    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in video_config and "resized_width" in video_config:
        resized_height, resized_width = smart_resize(
            video_config["resized_height"],
            video_config["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
    )
    if not is_cpu():
        video = video.pin_memory()
    return video, metadata


async def preprocess_video(
    vr,
    image_factor=IMAGE_FACTOR,
    video_config=None,
    *,
    video_processor=None,
    resize_raw_frames=False,
):
    return preprocess_video_sync(
        vr,
        image_factor=image_factor,
        video_config=video_config,
        video_processor=video_processor,
        processor_kwargs=video_config,
        resize_raw_frames=resize_raw_frames,
    )


# Compatible with Qwen-VL & Qwen-Omni Series
class QwenVLImageProcessor(SGLangBaseProcessor):
    supports_token_expansion = True
    supports_transformers_backend = True
    position_encoding = "qwen"
    # Non-Qwen subclasses retain their existing token layouts.
    _TOKENIZED_MEDIA_MODEL_TYPES = frozenset(
        {
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_omni_moe",
            "qwen4_exp",
        }
    )
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5ForCausalLMMTP,
        InternS2PreviewForConditionalGeneration,
        InternS2MobiusForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
        Cosmos3ForConditionalGeneration,
        Qwen4ExpForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type
        self.supports_token_expansion = (
            self.supports_token_expansion
            and self.model_type in self._TOKENIZED_MEDIA_MODEL_TYPES
        )
        self.prefer_tokenized_input = self.supports_token_expansion
        if self.model_type in (
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen4_exp",
            "intern_s2_preview",
            "interns2_mobius",
        ):
            # Two workers overlap CPU preprocessing without over-fragmenting
            # burst arrivals into smaller GPU prefill batches. Higher counts can
            # improve short-output TTFT, but regress long-output throughput on
            # Blackwell when requests reach the scheduler too far apart.
            self.auto_mm_processor_worker_num = 2
            self.auto_mm_io_worker_num = 16
            self.supports_mm_processor_concurrency = True
        if hf_config.model_type == "qwen3_omni_moe":
            self.media_processor_kwargs_type = Qwen3OmniMoeProcessorKwargs
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id

        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)

        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

        self._spatial_merge_size = self.hf_config.vision_config.spatial_merge_size
        self._tokens_per_second = getattr(
            self.hf_config.vision_config, "tokens_per_second", None
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=self.VIDEO_TOKEN_ID,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    @property
    def spatial_merge_size(self):
        return self._spatial_merge_size

    def build_input_ids_with_timestamps(
        self, prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
    ):
        """
        Build input_ids with timestamps for qwen3_vl models.
        """
        if not isinstance(prompt, list):
            prompt = self._processor.tokenizer.encode(prompt)

        img_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        spatial_merge_size = self.spatial_merge_size
        vision_start_token_id = getattr(self, "vision_start_token_id", None)
        vision_end_token_id = getattr(self, "vision_end_token_id", None)

        input_ids = []
        offsets = []
        modality_list = []
        cur_idx = 0

        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if img_token_id is not None and prompt[i + 1] == img_token_id:
                vision_start_indices.append((i, Modality.IMAGE))
            elif video_token_id is not None and prompt[i + 1] == video_token_id:
                vision_start_indices.append((i, Modality.VIDEO))

        img_idx = 0
        video_idx = 0
        for mm_start_idx, modality in vision_start_indices:
            modality_list.append(modality)
            video_tokens = None
            if modality == Modality.IMAGE:
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)
                mm_token_id = img_token_id
                img_idx += 1
            elif modality == Modality.VIDEO:
                curr_timestamps = video_timestamps[video_idx]
                num_frames = video_grid_thw[video_idx][0]
                frame_seqlen = video_grid_thw[video_idx][1:].prod().item() // (
                    spatial_merge_size**2
                )
                video_tokens = []
                _current_offset = len(input_ids) + mm_start_idx + 1 - cur_idx
                # take single frame as one mm_item
                for frame_idx in range(num_frames):
                    if frame_idx > 0:
                        modality_list.append(Modality.VIDEO)
                    curr_time = curr_timestamps[frame_idx]
                    timestamp_text = f"<{curr_time:.1f} seconds>"
                    timestamp_tokens = self._processor.tokenizer.encode(
                        timestamp_text, add_special_tokens=False
                    )
                    video_tokens.extend(timestamp_tokens)
                    _current_offset += len(timestamp_tokens)
                    if vision_start_token_id is not None:
                        video_tokens.append(vision_start_token_id)
                        _current_offset += 1
                    video_tokens.extend([video_token_id] * frame_seqlen)
                    if vision_end_token_id is not None:
                        video_tokens.append(vision_end_token_id)
                    offsets.append(
                        (_current_offset, _current_offset + frame_seqlen - 1)
                    )
                    _current_offset += (
                        frame_seqlen + 1
                        if vision_end_token_id is not None
                        else frame_seqlen
                    )  # for vision_end_token_id
                mm_token_num = len(video_tokens)
                mm_token_id = None
                video_idx += 1
            else:
                logger.warning(
                    f"{modality} modality is not supported for qwen3_vl models with timestamps."
                )
                continue
            assert cur_idx <= mm_start_idx
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            if modality == Modality.VIDEO:
                input_ids.extend(video_tokens)
            else:
                mm_offset_start = len(input_ids)
                input_ids.extend([mm_token_id] * mm_token_num)
                offsets.append((mm_offset_start, len(input_ids) - 1))
            cur_idx = mm_start_idx + 2  # jump to vision_end_id
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    def compute_mrope_positions(self, input_ids, mm_items):
        image_grid_thw = self._concat_mm_item_grid(
            mm_items, "image_grid_thw", Modality.IMAGE
        )
        video_grid_thw = self._concat_mm_item_grid(
            mm_items, "video_grid_thw", Modality.VIDEO
        )

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self._spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=self._tokens_per_second,
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    @staticmethod
    def _get_processor_output_value(ret, key):
        if ret is None:
            return None
        return ret.get(key) if hasattr(ret, "get") else getattr(ret, key, None)

    def _get_precomputed_mrope_from_output(self, ret):
        mrope_positions = self._get_processor_output_value(ret, "mrope_positions")
        mrope_position_delta = self._get_processor_output_value(
            ret, "mrope_position_delta"
        )
        if mrope_positions is None or mrope_position_delta is None:
            return None

        mrope_positions = torch.as_tensor(mrope_positions)
        if mrope_positions.ndim == 3:
            if mrope_positions.shape[1] != 1:
                return None
            mrope_positions = mrope_positions.squeeze(1)
        if mrope_positions.ndim != 2 or mrope_positions.shape[0] != 3:
            return None

        mrope_position_delta = torch.as_tensor(mrope_position_delta)
        if mrope_position_delta.ndim <= 1:
            mrope_position_delta = mrope_position_delta.reshape(-1, 1)
        return mrope_positions, mrope_position_delta

    @staticmethod
    def _as_grid_batch(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0) if value.ndim == 1 else value
        tensor = torch.as_tensor(value, dtype=torch.long)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    def _compute_image_only_mrope_positions_from_offsets(
        self,
        input_len: int,
        mm_items: List[MultimodalDataItem],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """instead of calling get_rope_index, build mrope position from mm_items.offsets and image_grid_thw of each image
        basically a simplified version of get_rope_index for image-only reqs
        """
        if self.model_type not in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen4_exp",
            "intern_s2_preview",
            "interns2_mobius",
            "cosmos3_omni",
        ):
            return None

        image_items = [item for item in mm_items if item.is_image()]
        if not image_items or len(image_items) != len(mm_items):
            return None

        spatial_merge_size = self._spatial_merge_size
        sorted_items = sorted(image_items, key=lambda item: item.offsets[0][0])
        position_segments = []
        st = 0
        next_pos = 0

        for item in sorted_items:
            if item.offsets is None or len(item.offsets) != 1:
                return None

            start, end = item.offsets[0]
            if start < st or end >= input_len:
                return None

            text_len = start - st
            if text_len > 0:
                position_segments.append(
                    torch.arange(text_len, dtype=dtype, device=device)
                    .view(1, -1)
                    .expand(3, -1)
                    + next_pos
                )
                next_pos += text_len

            grid = self._as_grid_batch(item.model_specific_data.get("image_grid_thw"))
            if grid is None or grid.shape[0] != 1:
                return None
            t, h, w = [int(x) for x in grid[0].tolist()]
            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w
            if num_image_tokens != end - start + 1:
                return None

            t_index = (
                torch.arange(llm_grid_t, dtype=dtype, device=device)
                .view(-1, 1)
                .expand(llm_grid_t, llm_grid_h * llm_grid_w)
                .reshape(-1)
            )
            h_index = (
                torch.arange(llm_grid_h, dtype=dtype, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            w_index = (
                torch.arange(llm_grid_w, dtype=dtype, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            position_segments.append(
                torch.stack([t_index, h_index, w_index]) + next_pos
            )
            next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)
            st = end + 1

        if st < input_len:
            text_len = input_len - st
            position_segments.append(
                torch.arange(text_len, dtype=dtype, device=device)
                .view(1, -1)
                .expand(3, -1)
                + next_pos
            )

        mrope_positions = torch.cat(position_segments, dim=1).unsqueeze(1)
        mrope_position_delta = (mrope_positions.max() + 1 - input_len).reshape(1, 1)
        return mrope_positions, mrope_position_delta

    @classmethod
    def _concat_mm_item_grid(cls, mm_items: list[MultimodalDataItem], key, modality):
        grids = []
        for item in mm_items:
            if not item.is_modality(modality):
                continue
            grid = cls._as_grid_batch(item.model_specific_data.get(key))
            if grid is not None:
                grids.append(grid)
        if not grids:
            return None
        if len(grids) == 1:
            return grids[0]
        return torch.cat(grids, dim=0)

    @classmethod
    def _get_grid_from_output_or_items(
        cls, ret, mm_items, key, modality, input_data=None
    ):
        grid = cls._get_processor_output_value(ret, key)
        if grid is None:
            grid = cls._concat_mm_item_grid(mm_items, key, modality)
        if grid is None and input_data and isinstance(input_data[0], dict):
            grid = input_data[0].get(key)
        return grid

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        audio_feature_lens = kwargs.get("audio_feature_lens", None)
        video_timestamps = kwargs.get("video_timestamps", None)
        second_per_grid_ts = kwargs.get("second_per_grid_ts", None)

        audio_seq_lens = None
        if audio_feature_lens is not None:
            if self.model_type == "qwen3_omni_moe":
                # apply _get_feat_extract_lengths to get seq_lens
                input_lengths_leave = audio_feature_lens % 100
                feat_lengths = (input_lengths_leave - 1) // 2 + 1
                audio_seq_lens = (
                    ((feat_lengths - 1) // 2 + 1 - 1) // 2
                    + 1
                    + (audio_feature_lens // 100) * 13
                )
            elif self.model_type == "qwen2_5_omni":
                audio_seq_lens = (audio_feature_lens - 1) // 2 + 1
                audio_seq_lens = (audio_seq_lens - 2) // 2 + 1

        if (
            self.model_type
            in [
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
                "qwen4_exp",
                "intern_s2_preview",
                "cosmos3_omni",
            ]
            and video_timestamps is not None
        ):
            input_ids, offsets, modality_list = self.build_input_ids_with_timestamps(
                prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
            )
        else:
            input_ids, offsets, modality_list = self.build_input_ids(
                prompt, img_grid_thw, video_grid_thw, audio_seq_lens=audio_seq_lens
            )
        assert all(isinstance(modality, Modality) for modality in modality_list)

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self._spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            image_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_audio_in_video=False,
            audio_seqlens=(
                audio_feature_lens if self.model_type == "qwen3_omni_moe" else None
            ),
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_token_id=self.audio_start_token_id,
            position_id_per_seconds=getattr(
                self.hf_config, "position_id_per_seconds", None
            ),
            tokens_per_second=self._tokens_per_second,
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_items = []
        consumed_per_modality = {}

        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            embedding_start = consumed_per_modality.get(modality, 0)
            embedding_slice = embeddings[modality][
                embedding_start : embedding_start + num_tokens
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    def process_videos(
        self, videos, processor, *, process_options=None, source_configs=None, **kwargs
    ):
        if kwargs.pop("use_audio_in_video", False):
            raise ValueError("Qwen token expansion does not support use_audio_in_video")
        kwargs.pop("seconds_per_chunk", None)
        kwargs.pop("position_id_per_seconds", None)
        supplied_metadata = kwargs.pop("video_metadata", None)
        video_processor = processor.video_processor
        native_video_resize = isinstance(video_processor, Qwen3VLVideoProcessor)
        recipes = process_options or [None] * len(videos)
        prepared, groups = [], []
        for index, video in enumerate(videos):
            config = {
                **self.video_config,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
                },
            }
            if source_configs:
                config.update(source_configs[index])
            processor_kwargs = dict(kwargs)
            recipe = recipes[index]
            if recipe:
                config = dict(recipe["video_config"])
                processor_kwargs = dict(recipe["processor_kwargs"])
            elif native_video_resize:
                processor_kwargs.update(
                    {
                        key: value
                        for key, value in config.items()
                        if key not in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
                    }
                )
                for key, value in {
                    "do_resize": video_processor.do_resize,
                    "size": video_processor.size,
                    "resample": video_processor.resample,
                    "patch_size": video_processor.patch_size,
                    "merge_size": video_processor.merge_size,
                    "temporal_patch_size": video_processor.temporal_patch_size,
                }.items():
                    processor_kwargs.setdefault(key, value)
                processor_kwargs["size"] = dict(processor_kwargs["size"])
                config["do_resize"] = processor_kwargs["do_resize"]
            try:
                frames, metadata = preprocess_video_sync(
                    video,
                    video_config=config,
                    video_processor=video_processor,
                    processor_kwargs=processor_kwargs,
                )
            finally:
                if isinstance(video, VideoDecoderWrapper):
                    video.close()
            if metadata is None:
                if recipe:
                    metadata = recipe["video_metadata"]
                elif supplied_metadata is not None:
                    metadata = supplied_metadata[index]
            if metadata is not None:
                processor_kwargs = {
                    key: value
                    for key, value in processor_kwargs.items()
                    if key not in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS - {"fps"}
                }
                processor_kwargs["do_sample_frames"] = False
            if native_video_resize and isinstance(video, VideoDecoderWrapper):
                processor_kwargs["do_resize"] = False
                processor_kwargs["input_data_format"] = "channels_first"
            processor_kwargs["return_metadata"] = True
            prepared.append((frames, metadata, config, processor_kwargs))
            for group_kwargs, indices in groups:
                if group_kwargs == processor_kwargs:
                    indices.append(index)
                    break
            else:
                groups.append((processor_kwargs, [index]))
        items = [None] * len(videos)
        video_features = [None] * len(videos)
        for processor_kwargs, indices in groups:
            metadata = [prepared[index][1] for index in indices]
            output = dict(
                processor.video_processor(
                    [prepared[index][0] for index in indices],
                    video_metadata=metadata
                    if all(item is not None for item in metadata)
                    else None,
                    **processor_kwargs,
                )
            )
            if self.model_type == "qwen2_5_vl":
                output["second_per_grid_ts"] = [
                    processor.video_processor.temporal_patch_size / item.sampled_fps
                    for item in output["video_metadata"]
                ]
            elif self.model_type == "qwen3_omni_moe":
                output["video_second_per_grid"] = [
                    processor.video_processor.temporal_patch_size
                    / processor_kwargs.get("fps", 1.0)
                ] * len(indices)
            elif self.model_type != "qwen2_vl":
                output["video_metadata"] = [
                    replace(item, fps=24) if item.fps is None else item
                    for item in output["video_metadata"]
                ]
            for group_index, index in enumerate(indices):
                items[index] = ProcessedMediaItem(
                    media_id=("video", index),
                    metadata={
                        name: output[name][group_index]
                        for name in (
                            "video_grid_thw",
                            "video_metadata",
                            "second_per_grid_ts",
                            "video_second_per_grid",
                        )
                        if name in output
                    },
                )
            if len(groups) > 1:
                group_items = get_new_expanded_mm_items(
                    self.collect_mm_items_from_processor_output(output)
                )
                for index, item in zip(indices, group_items):
                    video_features[index] = item.feature
        for index, item in enumerate(items):
            frames, _, config, processor_kwargs = prepared[index]
            metadata = item.metadata["video_metadata"]
            config["frame_indices"] = [int(frame) for frame in metadata.frames_indices]
            if isinstance(videos[index], VideoDecoderWrapper):
                config.update(
                    resized_height=int(frames.shape[-2]),
                    resized_width=int(frames.shape[-1]),
                )
            item.effective_options = {
                "video_config": config,
                "processor_kwargs": {
                    key: str(value) if key == "device" else value
                    for key, value in processor_kwargs.items()
                },
                "video_metadata": {
                    "fps": metadata.fps,
                    "total_num_frames": metadata.total_num_frames,
                    "frames_indices": config["frame_indices"],
                },
            }
        if len(groups) > 1:
            output = {
                "pixel_values_videos": torch.cat(video_features),
                "video_grid_thw": torch.stack(
                    [item.metadata["video_grid_thw"] for item in items]
                ),
                "video_metadata": [item.metadata["video_metadata"] for item in items],
            }
            for name in ("second_per_grid_ts", "video_second_per_grid"):
                if name in items[0].metadata:
                    output[name] = [item.metadata[name] for item in items]
        return MediaProcessOutput(encoder_inputs=output, items=items)

    def process_audio(self, audios, processor, **kwargs):
        grouped = process_media_groups(audios, processor, self.process_audio, kwargs)
        if grouped is not None:
            features = grouped.encoder_inputs["input_features"]
            if isinstance(features, list):
                grouped.encoder_inputs["input_features"] = (
                    torch.nn.utils.rnn.pad_sequence(
                        [feature.T for feature in features], batch_first=True
                    ).transpose(1, 2)
                )
                grouped.encoder_inputs["feature_attention_mask"] = (
                    torch.nn.utils.rnn.pad_sequence(
                        grouped.encoder_inputs["feature_attention_mask"],
                        batch_first=True,
                    )
                )
                for index, item in enumerate(grouped.items):
                    for name in ("input_features", "feature_attention_mask"):
                        item.encoder_inputs[name] = grouped.encoder_inputs[name][
                            index : index + 1
                        ]
            return grouped
        output = dict(processor.feature_extractor(audios, **kwargs))
        output["feature_attention_mask"] = output.pop("attention_mask")
        audio_token_counts = _get_feat_extract_output_lengths(
            output["feature_attention_mask"].sum(-1)
        )
        return MediaProcessOutput(
            output,
            [
                ProcessedMediaItem(
                    media_id=("audio", index),
                    encoder_inputs={
                        key: value[index : index + 1] for key, value in output.items()
                    },
                    metadata={"token_count": int(token_count)},
                    effective_options={
                        key: str(value) if key == "device" else value
                        for key, value in kwargs.items()
                    },
                    feature_name="input_features",
                )
                for index, token_count in enumerate(audio_token_counts)
            ],
        )

    def get_mm_token_replacements(self, processor, processed_media):
        image_fragments, video_fragments, audio_fragments = [], [], []
        if "image" in processed_media:
            merge_length = processor.image_processor.merge_size**2
            image_fragments = [
                [
                    (
                        [self.mm_tokens.image_token_id]
                        * (int(item.metadata["image_grid_thw"].prod()) // merge_length),
                        item.media_id,
                    )
                ]
                for item in processed_media["image"].items
            ]
        if "video" in processed_media:
            merge_length = processor.video_processor.merge_size**2
            for item in processed_media["video"].items:
                grid = item.metadata["video_grid_thw"]
                if self.model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_omni_moe"}:
                    video_fragments.append(
                        [
                            (
                                [self.mm_tokens.video_token_id]
                                * (int(grid.prod()) // merge_length),
                                item.media_id,
                            )
                        ]
                    )
                    continue
                metadata = item.metadata["video_metadata"]
                frame_count, height, width = grid.tolist()
                timestamps = processor._calculate_timestamps(
                    list(metadata.frames_indices),
                    metadata.fps,
                    processor.video_processor.temporal_patch_size,
                )
                fragment = []
                for timestamp in timestamps[:frame_count]:
                    fragment.extend(
                        [
                            (
                                processor.tokenizer.encode(
                                    f"<{timestamp:.1f} seconds>",
                                    add_special_tokens=False,
                                ),
                                None,
                            ),
                            ([self.vision_start_token_id], None),
                            (
                                [self.mm_tokens.video_token_id]
                                * (height * width // merge_length),
                                item.media_id,
                            ),
                            ([self.vision_end_token_id], None),
                        ]
                    )
                video_fragments.append(fragment)
        if "audio" in processed_media:
            audio_fragments = [
                [
                    (
                        [self.mm_tokens.audio_token_id] * item.metadata["token_count"],
                        item.media_id,
                    )
                ]
                for item in processed_media["audio"].items
            ]
        replacements = [
            ([self.mm_tokens.image_token_id], image_fragments),
            ([self.mm_tokens.video_token_id], video_fragments),
        ]
        if self.mm_tokens.audio_token_id is not None:
            replacements.append(([self.mm_tokens.audio_token_id], audio_fragments))
        return replacements

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        video_data=None,
        audio_data=None,
        input_ids=None,
        **kwargs,
    ):
        if video_data is None and request_obj is not None:
            video_data = request_obj.video_data
        if self.supports_token_expansion:
            return await super().process_mm_data_async(
                image_data=image_data,
                input_text=input_text,
                input_ids=input_ids,
                request_obj=request_obj,
                video_data=video_data,
                audio_data=audio_data,
                **kwargs,
            )
        if input_ids is not None:
            input_text = input_ids
        entry_time = time.perf_counter()
        base_output = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
        )
        load_time = time.perf_counter()
        rid = getattr(request_obj, "rid", "anonymous_rid")

        video_processor = getattr(self._processor, "video_processor", None)
        video_processor_kwargs = dict(self.video_config)
        if isinstance(video_processor, Qwen3VLVideoProcessor):
            video_device = self.video_preprocessing_device
            if (
                video_device is None
                and isinstance(self._processor.image_processor, BaseImageProcessor)
                and not self.disable_fast_image_processor
            ):
                video_device = self._fast_image_processor_device(self._processor)
            if video_device is not None:
                video_processor_kwargs["device"] = video_device
        base_output.videos, video_metadata = await self.process_video_data_async(
            base_output.videos,
            partial(
                preprocess_video_sync,
                video_config=self.video_config,
                video_processor=video_processor,
                processor_kwargs=video_processor_kwargs,
                resize_raw_frames=True,
            ),
        )
        processor_kwargs = {}
        processor_video_config = _get_processor_video_config(
            self.video_config, video_metadata
        )
        if processor_video_config is not None:
            if isinstance(video_processor, Qwen3VLVideoProcessor):
                processor_video_config["do_resize"] = False
                processor_video_config["input_data_format"] = "channels_first"
            processor_kwargs["processor_video_config"] = processor_video_config

        if isinstance(
            video_processor, Qwen3VLVideoProcessor
        ) or self.hf_config.model_type in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen4_exp",
            "intern_s2_preview",
            "interns2_mobius",
            "cosmos3_omni",
        ):
            processor_kwargs.update(
                video_metadata=video_metadata,
                do_sample_frames=False,
            )

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens, **processor_kwargs
        )

        self._mark_dp_encoder_features_for_deferred_reconstruction(mm_items)

        audio_feature_lengths = None

        if self.model_type == "qwen3_omni_moe":
            audio_item = next((mm for mm in mm_items if mm.is_audio()), None)
            if audio_item:
                audio_feature_lengths = torch.sum(
                    audio_item.feature_attention_mask, dim=1
                )

        second_per_grid_ts = self._get_processor_output_value(ret, "second_per_grid_ts")
        if second_per_grid_ts is None:
            second_per_grid_ts = self._get_processor_output_value(
                ret, "video_second_per_grid"
            )

        process_time = time.perf_counter()

        input_ids = input_ids.flatten()
        base_input_ids = getattr(base_output, "input_ids", None)
        if (
            isinstance(base_input_ids, list)
            and len(base_input_ids) == input_ids.numel()
        ):
            # reuse preprocess input if it already carries list of input_ids
            input_ids_list = base_input_ids
        else:
            input_ids_list = input_ids.tolist()

        # look for if padded_input_ids already exists before computing
        padded_input_ids = self._get_processor_output_value(ret, "padded_input_ids")
        if padded_input_ids is None:
            padded_input_ids = MultimodalProcessorOutput.build_padded_input_ids(
                input_ids_list, mm_items
            )
        elif isinstance(padded_input_ids, torch.Tensor):
            # reuse existing padded_input_ids
            padded_input_ids = padded_input_ids.flatten().tolist()
        else:
            padded_input_ids = list(padded_input_ids)

        image_grid_thw = self._get_grid_from_output_or_items(
            ret, mm_items, "image_grid_thw", Modality.IMAGE, image_data
        )
        video_grid_thw = self._get_grid_from_output_or_items(
            ret,
            mm_items,
            "video_grid_thw",
            Modality.VIDEO,
            video_data,
        )

        mrope_result = self._get_precomputed_mrope_from_output(ret)
        if mrope_result is None:
            if (
                video_grid_thw is None
                and second_per_grid_ts is None
                and audio_feature_lengths is None
            ):
                mrope_result = self._compute_image_only_mrope_positions_from_offsets(
                    input_len=input_ids.numel(),
                    mm_items=mm_items,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
        if mrope_result is None:
            mrope_result = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=self._spatial_merge_size,
                image_token_id=self.mm_tokens.image_token_id,
                video_token_id=self.mm_tokens.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type=self.model_type,
                tokens_per_second=self._tokens_per_second,
                # use the expanded token ids
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                use_audio_in_video=False,
                audio_seqlens=audio_feature_lengths,
                audio_token_id=getattr(self.hf_config, "audio_token_id", None),
                audio_start_token_id=self.audio_start_token_id,
                position_id_per_seconds=getattr(
                    self.hf_config, "position_id_per_seconds", None
                ),
            )

        mrope_positions, mrope_position_delta = mrope_result
        if mrope_positions.ndim == 3:
            mrope_positions = mrope_positions.squeeze(1)
        get_rope_index_time = time.perf_counter()
        logger.debug(
            f"[QwenVLProcessor Perf] {rid=}, "
            f"load_time: {(load_time - entry_time) * 1000:.2f} ms, "
            f"process_time: {(process_time - load_time) * 1000:.2f} ms, "
            f"get_rope_index_time: {(get_rope_index_time - process_time) * 1000:.2f} ms, "
            f"total_time: {(get_rope_index_time - entry_time) * 1000:.2f} ms"
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids_list,
            padded_input_ids=padded_input_ids,
            mm_items=mm_items,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    def _mark_dp_encoder_features_for_deferred_reconstruction(self, mm_items):
        if not (
            self.keep_mm_features_on_device
            and self.runtime_context.config_bag("mm").mm_enable_dp_encoder
            and self.model_type
            in ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe")
        ):
            return
        for item in mm_items:
            if item.is_image() or item.is_video():
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )
