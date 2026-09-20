import logging
import math
import os
import re
import time
from dataclasses import asdict, replace

import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode
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
from sglang.srt.multimodal.media_processing import process_media_groups
from sglang.srt.multimodal.media_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.token_space_processor import (
    TokenSpaceMultimodalProcessor,
)

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
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


# process video, qwen-specific
def preprocess_video_sync(
    vr,
    *,
    image_factor: int = IMAGE_FACTOR,
    video_config: dict = None,
    video_processor=None,
    processor_kwargs=None,
    resize_raw_frames=False,
):
    from sglang.srt.utils import is_cpu
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

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
    # preprocessed video
    is_video_obj = isinstance(vr, VideoDecoderWrapper)
    if not is_video_obj:
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
    entry_time = time.perf_counter()

    total_frames, video_fps = len(vr), vr.avg_fps

    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)

    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx,
        "video_backend": "torchvision",
    }
    if isinstance(video_processor, Qwen3VLVideoProcessor):
        return (
            _decode_native_video(
                vr, idx, video_processor, video_config, processor_kwargs
            ),
            video_metadata,
        )
    video = vr.get_frames_as_tensor(idx.tolist())

    video = video.permute(0, 3, 1, 2)  # NHWC -> TCHW

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

    get_batch_time = time.perf_counter()

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
    smart_resize_time = time.perf_counter()
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
    )
    if not is_cpu():
        video = video.pin_memory()
    torchvision_resize_time = time.perf_counter()
    logger.debug(
        f"[preprocess_video Perf], "
        f"get_batch_time: {(get_batch_time - entry_time) * 1000:.2f} ms, "
        f"smart_resize_time: {(smart_resize_time - get_batch_time) * 1000:.2f} ms, "
        f"torchvision_resize_time: {(torchvision_resize_time - smart_resize_time) * 1000:.2f} ms, "
        f"total_time: {(torchvision_resize_time - entry_time) * 1000:.2f} ms"
    )
    return video, video_metadata


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class QwenTokenSpaceProcessor(TokenSpaceMultimodalProcessor):
    supports_token_expansion = True
    supports_transformers_backend = True
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

    @classmethod
    def supports_token_space_processing(cls, hf_config):
        return (
            cls.supports_token_expansion
            and hf_config.model_type in cls._TOKENIZED_MEDIA_MODEL_TYPES
        )

    def __init__(self, hf_config, processor, **kwargs):
        self.model_type = hf_config.model_type
        self.supports_token_expansion = self.supports_token_space_processing(hf_config)
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
        if self.model_type == "qwen3_omni_moe":
            self.media_processor_kwargs_type = Qwen3OmniMoeProcessorKwargs
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, processor, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.IM_TOKEN_ID = self.image_token_id = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = self.video_token_id = hf_config.video_token_id

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
        ).build(processor)

    def process_videos(self, videos, processor, *, source_configs=None, **kwargs):
        from sglang.srt.utils.video_decoder import VideoDecoderWrapper

        if kwargs.pop("use_audio_in_video", False):
            raise ValueError("Qwen token expansion does not support use_audio_in_video")
        kwargs.pop("seconds_per_chunk", None)
        kwargs.pop("position_id_per_seconds", None)
        supplied_metadata = kwargs.pop("video_metadata", None)
        video_processor = processor.video_processor
        native_video_resize = isinstance(video_processor, Qwen3VLVideoProcessor)
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
            if native_video_resize:
                processor_kwargs.update(
                    {
                        key: value
                        for key, value in config.items()
                        if key not in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS
                    }
                )
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
            if metadata is None and supplied_metadata is not None:
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
            prepared.append((frames, metadata))
            for group_kwargs, indices in groups:
                if group_kwargs == processor_kwargs:
                    indices.append(index)
                    break
            else:
                groups.append((processor_kwargs, [index]))
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
            if len(groups) > 1:
                for index, features in zip(
                    indices, self.split_media_features(Modality.VIDEO, output)
                ):
                    video_features[index] = features
        if len(groups) > 1:
            output = self.merge_media_features(Modality.VIDEO, video_features)
        return output

    def process_audio(self, audios, processor, **kwargs):
        grouped = process_media_groups(audios, processor, self.process_audio, kwargs)
        if grouped is not None:
            features = grouped["input_features"]
            if isinstance(features, list):
                grouped["input_features"] = torch.nn.utils.rnn.pad_sequence(
                    [feature.T for feature in features], batch_first=True
                ).transpose(1, 2)
                grouped["feature_attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                    grouped["feature_attention_mask"], batch_first=True
                )
            return grouped
        output = dict(processor.feature_extractor(audios, **kwargs))
        output["feature_attention_mask"] = output.pop("attention_mask")
        return output

    def split_media_features(self, modality: Modality, features: dict) -> list[dict]:
        if modality == Modality.AUDIO:
            return [
                {name: value[index : index + 1] for name, value in features.items()}
                for index in range(len(features["input_features"]))
            ]
        feature_name, grid_name = {
            Modality.IMAGE: ("pixel_values", "image_grid_thw"),
            Modality.VIDEO: ("pixel_values_videos", "video_grid_thw"),
        }[modality]
        patch_counts = features[grid_name].prod(-1).tolist()
        return [
            {
                name: patches if name == feature_name else value[index : index + 1]
                for name, value in features.items()
            }
            for index, patches in enumerate(features[feature_name].split(patch_counts))
        ]

    def merge_media_features(self, modality: Modality, features: list[dict]) -> dict:
        if modality == Modality.AUDIO:
            return {
                "input_features": torch.nn.utils.rnn.pad_sequence(
                    [feature["input_features"][0].T for feature in features],
                    batch_first=True,
                ).transpose(1, 2),
                "feature_attention_mask": torch.nn.utils.rnn.pad_sequence(
                    [feature["feature_attention_mask"][0] for feature in features],
                    batch_first=True,
                ),
            }
        return {
            name: (
                torch.cat([feature[name] for feature in features])
                if isinstance(features[0][name], torch.Tensor)
                else [value for feature in features for value in feature[name]]
            )
            for name in features[0]
        }

    def get_mm_token_expansion_spec(self, processor, media_features):
        image_expansions, video_expansions, audio_expansions = [], [], []
        if "image_grid_thw" in media_features:
            merge_length = processor.image_processor.merge_size**2
            image_expansions = [
                [self.image_token_id] * (int(grid.prod()) // merge_length)
                for grid in media_features["image_grid_thw"]
            ]
        if "video_grid_thw" in media_features:
            merge_length = processor.video_processor.merge_size**2
            for index, grid in enumerate(media_features["video_grid_thw"]):
                if self.model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_omni_moe"}:
                    video_expansions.append(
                        [self.video_token_id] * (int(grid.prod()) // merge_length)
                    )
                    continue
                metadata = media_features["video_metadata"][index]
                frame_count, height, width = grid.tolist()
                timestamps = processor._calculate_timestamps(
                    list(metadata.frames_indices),
                    metadata.fps,
                    processor.video_processor.temporal_patch_size,
                )
                expanded_tokens = []
                for timestamp in timestamps[:frame_count]:
                    expanded_tokens.extend(
                        processor.tokenizer.encode(
                            f"<{timestamp:.1f} seconds>", add_special_tokens=False
                        )
                        + [self.vision_start_token_id]
                        + [self.video_token_id] * (height * width // merge_length)
                        + [self.vision_end_token_id]
                    )
                video_expansions.append(expanded_tokens)
        if "feature_attention_mask" in media_features:
            audio_token_counts = _get_feat_extract_output_lengths(
                media_features["feature_attention_mask"].sum(-1)
            )
            audio_expansions = [
                [self.audio_token_id] * int(token_count)
                for token_count in audio_token_counts
            ]
        mm_token_expansion_spec = [
            ([self.image_token_id], image_expansions),
            ([self.video_token_id], video_expansions),
        ]
        if self.audio_token_id is not None:
            mm_token_expansion_spec.append(([self.audio_token_id], audio_expansions))
        return mm_token_expansion_spec

    @property
    def spatial_merge_size(self):
        return self._spatial_merge_size
