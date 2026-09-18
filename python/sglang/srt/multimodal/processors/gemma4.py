# Copyright 2025 SGLang Team
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

import re
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers.models.gemma4.image_processing_gemma4 import pad_along_first_dim
from transformers.models.gemma4.video_processing_gemma4 import pad_to_max_patches

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.gemma4_audio import _SSCP_CONV_STRIDE_SIZES
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.multimodal.media_processing import (
    MediaProcessOutput,
    ProcessedMediaItem,
    process_media_groups,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.utils.video_decoder import VideoDecoderWrapper


def _video_metadata_as_dict(metadata):
    metadata = dict(metadata)
    return {
        "total_num_frames": int(metadata["total_num_frames"]),
        "frames_indices": [int(index) for index in metadata["frames_indices"]],
        **{
            key: float(metadata[key]) if metadata[key] is not None else None
            for key in ("fps", "duration")
            if key in metadata
        },
    }


def _collate_vision_media(output, feature_name, position_name):
    if isinstance(output.encoder_inputs.get(feature_name), torch.Tensor):
        return output
    pixels = [item.encoder_inputs[feature_name][0] for item in output.items]
    positions = [item.encoder_inputs[position_name][0] for item in output.items]
    max_patches = max(source_pixels.shape[-2] for source_pixels in pixels)
    is_video = feature_name == "pixel_values_videos"
    pad_patches = pad_to_max_patches if is_video else pad_along_first_dim
    padded_inputs = [
        pad_patches(source_pixels, source_positions, max_patches)
        for source_pixels, source_positions in zip(pixels, positions)
    ]
    if is_video:
        # Encode only real frames; an all-padding frame can create a pooled token.
        output.encoder_inputs[feature_name] = torch.cat(
            [source_pixels for source_pixels, _ in padded_inputs]
        ).unsqueeze(0)
        output.encoder_inputs[position_name] = torch.cat(
            [source_positions for _, source_positions in padded_inputs]
        ).unsqueeze(0)
        frame_start = 0
        for item, source_pixels in zip(output.items, pixels):
            frame_end = frame_start + len(source_pixels)
            original_patch_count = source_pixels.shape[-2]
            for key in (feature_name, position_name):
                item.encoder_inputs[key] = output.encoder_inputs[key][
                    :, frame_start:frame_end, :original_patch_count
                ]
            frame_start = frame_end
    else:
        output.encoder_inputs[feature_name] = torch.stack(
            [source_pixels for source_pixels, _ in padded_inputs]
        )
        output.encoder_inputs[position_name] = torch.stack(
            [source_positions for _, source_positions in padded_inputs]
        )
        for index, (item, source_pixels) in enumerate(zip(output.items, pixels)):
            original_patch_count = source_pixels.shape[-2]
            for key in (feature_name, position_name):
                item.encoder_inputs[key] = output.encoder_inputs[key][
                    index : index + 1, :original_patch_count
                ]
    return output


def _collate_audio_media(output):
    for key in ("input_features", "input_features_mask"):
        if not isinstance(output.encoder_inputs[key], torch.Tensor):
            output.encoder_inputs[key] = torch.nn.utils.rnn.pad_sequence(
                [item.encoder_inputs[key][0] for item in output.items],
                batch_first=True,
                padding_value=0,
            )
            for index, item in enumerate(output.items):
                original_frame_count = item.encoder_inputs[key].shape[1]
                item.encoder_inputs[key] = output.encoder_inputs[key][
                    index : index + 1, :original_frame_count
                ]
    return output


class Gemma4SGLangProcessor(SGLangBaseProcessor):
    """Multimodal processor for Gemma4 supporting image, video, and audio inputs."""

    models = [Gemma4ForConditionalGeneration]
    supports_token_expansion = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.prefer_tokenized_input = self.supports_token_expansion
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.boi_token_id
        self.IM_END_TOKEN_ID = hf_config.eoi_token_id

        self.AUDIO_START_TOKEN_ID = hf_config.boa_token_id
        self.AUDIO_END_TOKEN_ID = hf_config.eoa_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|image>(?:<\|image\|>)+<image\|>|<\|image\|>"
            ),
            video_token="<|video|>",
            video_token_id=hf_config.video_token_id,
            video_token_regex=re.compile(
                r"<\|image>(?:<\|video\|>)+<image\|>|<\|video\|>"
            ),
            audio_token="<|audio|>",
            audio_token_id=hf_config.audio_token_id,
            audio_token_regex=re.compile(
                r"<\|audio>(?:<\|audio\|>)+<audio\|>|<\|audio\|>"
            ),
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE
        self.ATTR_NAME_TO_MODALITY["video_position_ids"] = Modality.VIDEO

    def _get_audio_pad_multiple(self, processor) -> int:
        return processor.feature_extractor.hop_length * _SSCP_CONV_STRIDE_SIZES[0][0]

    def _get_audio_token_count(self, mask) -> int:
        # Two stride-2 SSCP blocks retain every fourth mask entry.
        return int(mask[::4].sum())

    def _video_decoder_to_tensor(
        self, vdw: VideoDecoderWrapper, processor
    ) -> torch.Tensor:
        """Convert a VideoDecoderWrapper to a (sampled_frames, C, H, W) uint8 tensor.

        SGLang's load_video returns VideoDecoderWrapper which the HF
        Gemma4VideoProcessor does not recognise (expects torch.Tensor or
        np.ndarray).  We replicate HF's uniform frame sampling here to
        avoid materialising the entire video in memory, then delegate the
        rest (resize, patchify, position IDs) to the HF video processor.
        """
        total = len(vdw)
        num_frames = getattr(
            getattr(processor, "video_processor", None),
            "num_frames",
            32,
        )
        if total <= num_frames:
            indices = list(range(total))
        else:
            indices = torch.arange(0, total, total / num_frames).int().tolist()
        frames_np = vdw.get_frames_at(indices)  # (N, H, W, C)
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()

    def process_mm_data(
        self,
        input_text="",
        images=None,
        videos=None,
        audios=None,
        processor=None,
        *,
        input_ids=None,
        **kwargs,
    ):
        if self.supports_token_expansion:
            return super().process_mm_data(
                input_text,
                images=images,
                videos=videos,
                audios=audios,
                processor=processor,
                input_ids=input_ids,
                **kwargs,
            )
        processor, _ = self._resolve_processor(processor)
        if audios:
            pad_multiple = self._get_audio_pad_multiple(processor)
            padded = []
            for a in audios:
                a = np.asarray(a)
                remainder = len(a) % pad_multiple
                if remainder != 0:
                    a = np.pad(a, (0, pad_multiple - remainder), mode="constant")
                padded.append(a)
            audios = padded
        if videos:
            videos = [
                (
                    self._video_decoder_to_tensor(v, processor)
                    if isinstance(v, VideoDecoderWrapper)
                    else v
                )
                for v in videos
            ]
            kwargs.setdefault("do_sample_frames", False)
        return super().process_mm_data(
            input_text,
            images=images,
            videos=videos,
            audios=audios,
            processor=processor,
            input_ids=input_ids,
            **kwargs,
        )

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        video_data=None,
        input_ids=None,
        **kwargs,
    ):
        if video_data is None and request_obj is not None:
            video_data = request_obj.video_data
        if self.supports_token_expansion:
            return await super().process_mm_data_async(
                image_data=image_data,
                audio_data=audio_data,
                input_text=input_text,
                input_ids=input_ids,
                request_obj=request_obj,
                video_data=video_data,
                **kwargs,
            )
        if input_ids is not None:
            input_text = input_ids
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
        )

    def process_images(self, images, processor, **kwargs):
        grouped = process_media_groups(images, processor, self.process_images, kwargs)
        if grouped is not None:
            return _collate_vision_media(grouped, "pixel_values", "image_position_ids")
        output = dict(processor.image_processor(images, **kwargs))
        token_counts = output.pop("num_soft_tokens_per_image")
        return MediaProcessOutput(
            encoder_inputs=output,
            items=[
                ProcessedMediaItem(
                    media_id=("image", index),
                    encoder_inputs={
                        key: value[index : index + 1] for key, value in output.items()
                    },
                    metadata={"num_soft_tokens": int(token_count)},
                    effective_options={
                        key: str(value) if isinstance(value, torch.device) else value
                        for key, value in kwargs.items()
                    },
                    feature_name="pixel_values",
                )
                for index, token_count in enumerate(token_counts)
            ],
        )

    def _sample_video_frames(self, video, num_frames, metadata=None):
        if not isinstance(video, VideoDecoderWrapper):
            frame_count = (
                1
                if isinstance(video, (torch.Tensor, np.ndarray)) and video.ndim == 3
                else len(video)
            )
            return video, {
                "total_num_frames": frame_count,
                "frames_indices": list(range(frame_count)),
            }
        total_frames = len(video)
        frame_indices = (
            metadata["frames_indices"]
            if metadata is not None
            else (
                list(range(total_frames))
                if total_frames <= num_frames
                else torch.arange(0, total_frames, total_frames / num_frames)
                .int()
                .tolist()
            )
        )
        frames = torch.as_tensor(video.get_frames_at(frame_indices))
        return frames.permute(0, 3, 1, 2).contiguous(), {
            "fps": video.avg_fps,
            "duration": total_frames / video.avg_fps,
            "total_num_frames": total_frames,
            "frames_indices": frame_indices,
        }

    def process_videos(self, videos, processor, **kwargs):
        recipes = kwargs.pop("process_options", None)
        source_configs = kwargs.pop("source_configs", None)
        if recipes is not None or source_configs is not None:
            items = []
            for index, video in enumerate(videos):
                recipe = recipes[index] if recipes is not None else None
                if recipe is not None:
                    options = dict(recipe)
                else:
                    options = dict(kwargs)
                    metadata = options.get("video_metadata")
                    if metadata is not None and not isinstance(metadata, dict):
                        options["video_metadata"] = [metadata[index]]
                    if source_configs is not None:
                        options.update(source_configs[index])
                output = self.process_videos([video], processor, **options)
                item = output.items[0]
                item.media_id = ("video", index)
                items.append(item)
            return _collate_vision_media(
                MediaProcessOutput(encoder_inputs={}, items=items),
                "pixel_values_videos",
                "video_position_ids",
            )
        sampled_videos = []
        video_metadata = []
        num_frames = kwargs.get("num_frames", processor.video_processor.num_frames)
        supplied_metadata = kwargs.pop("video_metadata", None)
        if isinstance(supplied_metadata, dict):
            supplied_metadata = [supplied_metadata]
        if supplied_metadata is not None:
            supplied_metadata = [
                _video_metadata_as_dict(item) for item in supplied_metadata
            ]
        for index, video in enumerate(videos):
            try:
                frames, metadata = self._sample_video_frames(
                    video,
                    num_frames,
                    supplied_metadata[index] if supplied_metadata is not None else None,
                )
                sampled_videos.append(frames)
                video_metadata.append(
                    supplied_metadata[index]
                    if supplied_metadata is not None
                    else metadata
                )
            finally:
                if isinstance(video, VideoDecoderWrapper):
                    video.close()
        kwargs["do_sample_frames"] = False
        kwargs["return_metadata"] = True
        frame_counts = [
            1
            if isinstance(video, (torch.Tensor, np.ndarray)) and video.ndim == 3
            else len(video)
            for video in sampled_videos
        ]
        if len(set(frame_counts)) > 1:
            items = []
            group_start = 0
            while group_start < len(sampled_videos):
                group_end = group_start + 1
                while (
                    group_end < len(sampled_videos)
                    and frame_counts[group_end] == frame_counts[group_start]
                ):
                    group_end += 1
                group = self.process_videos(
                    sampled_videos[group_start:group_end],
                    processor,
                    video_metadata=video_metadata[group_start:group_end],
                    **kwargs,
                )
                for index, item in enumerate(group.items, start=group_start):
                    item.media_id = ("video", index)
                    items.append(item)
                group_start = group_end
            return _collate_vision_media(
                MediaProcessOutput(encoder_inputs={}, items=items),
                "pixel_values_videos",
                "video_position_ids",
            )
        output = dict(
            processor.video_processor(
                sampled_videos, video_metadata=video_metadata, **kwargs
            )
        )
        token_counts = output.pop("num_soft_tokens_per_video")
        metadata = output.pop("video_metadata")
        items = []
        for index, (token_count, timeline) in enumerate(zip(token_counts, metadata)):
            fps = timeline.fps if timeline.fps is not None else 24
            items.append(
                ProcessedMediaItem(
                    media_id=("video", index),
                    encoder_inputs={
                        key: value[index : index + 1] for key, value in output.items()
                    },
                    metadata={
                        "num_soft_tokens": int(token_count),
                        "timestamps": [
                            frame_index / fps for frame_index in timeline.frames_indices
                        ],
                    },
                    effective_options={
                        **{
                            key: str(value)
                            if isinstance(value, torch.device)
                            else value
                            for key, value in kwargs.items()
                        },
                        "num_frames": num_frames,
                        "video_metadata": _video_metadata_as_dict(
                            video_metadata[index]
                        ),
                    },
                    feature_name="pixel_values_videos",
                )
            )
        return MediaProcessOutput(encoder_inputs=output, items=items)

    def process_audio(self, audios, processor, **kwargs):
        grouped = process_media_groups(audios, processor, self.process_audio, kwargs)
        if grouped is not None:
            return _collate_audio_media(grouped)
        kwargs.setdefault("sampling_rate", processor.feature_extractor.sampling_rate)
        pad_multiple = self._get_audio_pad_multiple(processor)
        waveforms = []
        for audio in audios:
            audio = np.asarray(audio)
            remainder = len(audio) % pad_multiple
            if remainder:
                audio = np.pad(audio, (0, pad_multiple - remainder), mode="constant")
            waveforms.append(audio)
        kwargs.setdefault("truncation", False)
        output = dict(processor.feature_extractor(waveforms, **kwargs))
        return MediaProcessOutput(
            encoder_inputs=output,
            items=[
                ProcessedMediaItem(
                    media_id=("audio", index),
                    encoder_inputs={
                        key: value[index : index + 1] for key, value in output.items()
                    },
                    metadata={"num_audio_tokens": self._get_audio_token_count(mask)},
                    effective_options={
                        key: str(value) if isinstance(value, torch.device) else value
                        for key, value in kwargs.items()
                    },
                    feature_name="input_features",
                )
                for index, mask in enumerate(output["input_features_mask"])
            ],
        )

    def get_mm_token_replacements(self, processor, processed_media):
        replacements = []
        for modality, placeholder_id, start_id, end_id in (
            (
                "image",
                self.mm_tokens.image_token_id,
                self.IM_START_TOKEN_ID,
                self.IM_END_TOKEN_ID,
            ),
            (
                "video",
                self.mm_tokens.video_token_id,
                self.IM_START_TOKEN_ID,
                self.IM_END_TOKEN_ID,
            ),
            (
                "audio",
                self.mm_tokens.audio_token_id,
                self.AUDIO_START_TOKEN_ID,
                self.AUDIO_END_TOKEN_ID,
            ),
        ):
            media_output = processed_media.get(modality)
            items = media_output.items if media_output is not None else []
            if placeholder_id is None:
                if items:
                    raise ValueError(
                        "Gemma4 media input has no configured placeholder ID"
                    )
                continue
            fragments = []
            for item in items:
                token_count = item.metadata[
                    "num_audio_tokens" if modality == "audio" else "num_soft_tokens"
                ]
                block = [
                    ([start_id], None),
                    ([placeholder_id] * token_count, item.media_id),
                    ([end_id], None),
                ]
                fragment = []
                if modality == "video":
                    for frame_index, seconds in enumerate(item.metadata["timestamps"]):
                        timestamp = (
                            " " if frame_index else ""
                        ) + f"{int(seconds // 60):02d}:{int(seconds % 60):02d} "
                        fragment.append(
                            (
                                processor.tokenizer.encode(
                                    timestamp, add_special_tokens=False
                                ),
                                None,
                            )
                        )
                        fragment.extend(block)
                else:
                    fragment = block
                fragments.append(fragment)
            replacements.append(([placeholder_id], fragments))
        return replacements
