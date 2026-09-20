import asyncio
import concurrent.futures
import dataclasses
import multiprocessing as mp
import re
import threading
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessor

from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat
from sglang.srt.utils import (
    CLIENT_MEDIA_EXCEPTIONS,
    ImageData,
    VideoData,
    load_audio,
    load_image,
    load_video,
    logger,
    smart_to_rgb,
)


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text with all multimodality placeholder token expanded
    input_text: str

    # original pre-tokenized ids, useful for processor_output/precomputed inputs,
    # when they already carry the input ids
    input_ids: Optional[Union[List[int], torch.Tensor]] = None

    # frames loaded from image, in given order
    images: Optional[list[Union[Image.Image, dict]]] = dataclasses.field(
        default_factory=list
    )

    # videos
    videos: Optional[list[Union[torch.Tensor, dict]]] = dataclasses.field(
        default_factory=list
    )

    # audios
    audios: Optional[list[Union[np.ndarray, dict]]] = dataclasses.field(
        default_factory=list
    )

    def organize_results(self) -> List[Tuple[Modality, Any]]:
        """

        :return: a list of results, with their corresponding modalities
        """
        return (
            [(Modality.IMAGE, data) for data in self.images]
            + [(Modality.VIDEO, data) for data in self.videos]
            + [(Modality.AUDIO, data) for data in self.audios]
        )


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[str, List[str]]] = None
    video_token: Optional[Union[str, List[str]]] = None
    audio_token: Optional[Union[str, List[str]]] = None

    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    combined_regex: Optional[re.Pattern] = None

    def build(self, processor):
        self.convert_to_strs(processor)
        self.parse_regex()
        self.get_combined_regex()
        return self

    def convert_to_str(self, token: Union[str, int], processor) -> str:
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor):
        if not self.image_token:
            self.image_token = self.convert_to_str(self.image_token_id, processor)
        if not self.video_token:
            self.video_token = self.convert_to_str(self.video_token_id, processor)
        if not self.audio_token:
            self.audio_token = self.convert_to_str(self.audio_token_id, processor)

    def get_modality_of_token(self, token: str) -> Optional[Modality]:
        """
        :return: the modality associated with the given token, if the token is a special_token or matches with the multimodal token regex
        """
        modality = {
            self.image_token: Modality.IMAGE,
            self.video_token: Modality.VIDEO,
            self.audio_token: Modality.AUDIO,
        }.get(token)
        if modality:
            return modality

        for regex, modality in [
            (self.image_token_regex, Modality.IMAGE),
            (self.video_token_regex, Modality.VIDEO),
            (self.audio_token_regex, Modality.AUDIO),
        ]:
            if regex and regex.match(token):
                return modality

        return None

    def get_token_id_by_modality(self, modality: Modality) -> Optional[int]:
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)

    def parse_regex(self):
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def get_combined_regex(self) -> re.Pattern:
        """
        Builds and returns a regex, used to split input str into tokens (with mm special tokens)
        """
        if self.combined_regex:
            return self.combined_regex
        tokens = [
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []
        flags = 0
        for t in tokens:
            if t is not None:
                patterns.append(t.pattern)
                flags |= t.flags
        combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"
        self.combined_regex = re.compile(combined, flags)
        return self.combined_regex


def _tokenizer_of(processor):
    """The tokenizer reached from an HF processor.

    Some processors (e.g. InternVL) are handed a tokenizer directly as their
    ``_processor`` rather than one that wraps a tokenizer. Every path that
    resolves a tokenizer -- construction and per-worker processor clones alike --
    goes through here, so a clone cannot resolve differently from the original.
    """
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


class MultimodalProcessorMixin:
    use_token_space_processor = False
    gpu_image_decode = True  # Enable GPU decoding by default
    smart_rgb_conversion = False
    video_preprocessing_device = None
    # None lets the worker count follow where preprocessing actually runs; a
    # model that measured its own optimum assigns a number instead. See
    # `_resolve_auto_mm_processor_worker_num`.
    auto_mm_processor_worker_num = None
    auto_mm_io_worker_num = 4
    # Processors opt out only when their preprocessing is not thread-safe. The
    # worker pool gives each thread its own `copy.deepcopy` of the HF processor
    # and injects it, and the single function it runs --
    # `process_and_combine_mm_data` -- resolves that clone instead of
    # `self._processor`, so isolation does not depend on the subclass.
    supports_mm_processor_concurrency = True

    def _create_cpu_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        return concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context(self.processor_config.cpu_process_start_method),
            max_workers=self.processor_config.cpu_worker_num,
        )

    def _replace_broken_cpu_executor(
        self, failed_executor: concurrent.futures.ProcessPoolExecutor
    ) -> None:
        """Replace a failed preprocess pool once across concurrent requests."""
        with self._cpu_executor_lock:
            if self.cpu_executor is not failed_executor:
                return
            self.cpu_executor = self._create_cpu_executor()
        logger.warning("Replaced a broken multimodal CPU preprocess pool")
        threading.Thread(
            target=self._shutdown_broken_cpu_executor,
            args=(failed_executor,),
            name="sglang-mm-cpu-pool-cleanup",
            daemon=True,
        ).start()

    @staticmethod
    def _shutdown_broken_cpu_executor(
        failed_executor: concurrent.futures.ProcessPoolExecutor,
    ) -> None:
        try:
            failed_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            logger.warning(
                "Failed to shut down a broken multimodal CPU preprocess pool",
                exc_info=True,
            )

    def _resolve_processor(self, processor=None):
        if processor is None:
            return self._processor, self._tokenizer
        return processor, _tokenizer_of(processor)

    def _preprocessing_competes_with_the_scheduler(self) -> bool:
        """Whether image preprocessing submits its work to the serving GPU.

        The fast image processor runs inside the tokenizer process but on
        ``cuda:{base_gpu_id}`` -- the device the scheduler serves from. A second
        preprocessing worker there is one more competitor for that device rather
        than added parallelism.
        """
        if self.processor_config.device in (None, "cpu"):
            return False
        if self.disable_fast_image_processor:
            return False
        image_processor = getattr(self._processor, "image_processor", None)
        return isinstance(image_processor, BaseImageProcessor)

    def _resolve_auto_mm_processor_worker_num(self) -> int:
        """The worker count to use when the user did not ask for one.

        Two workers overlap preprocessing that runs on the CPU, where the second
        thread is real parallelism: measured on Qwen2.5-VL with full-page images
        at 32-way concurrency, 4.46 -> 6.08 req/s on H200 and 7.07 -> 8.76 on
        GB300.

        The GPU path is capped at one worker even when a model declares more.
        A declaration records what its author measured on one platform and one
        image shape; contending for the device the scheduler is serving from is a
        property of the path itself, and it does not go away because a subclass
        asked for concurrency. Qwen-VL declares two and is the model that
        measures 9.30 -> 4.02 req/s on GB300 full-page images, so honouring the
        declaration here would exempt exactly the case that regresses.
        `--mm-processor-worker-num` still overrides this.
        """
        if self._preprocessing_competes_with_the_scheduler():
            return 1
        declared = self.auto_mm_processor_worker_num
        return 2 if declared is None else declared

    def _get_preprocessing_device(self) -> Optional[str]:
        return self.processor_config.device

    def _fast_image_processor_device(self, processor) -> Optional[str]:
        """The device for the fast image processor, or None to leave it unset."""
        device = self._get_preprocessing_device()
        if device != "npu":
            return device
        if processor.__class__.__name__ == "MiniMaxVLProcessor":
            # MiniMax's image/video processors create 10-dim tensors during
            # patch extraction, exceeding the Ascend 8-dim limit; patch them
            # (same pattern as qwen-vl / GLM-4.6V) and run on NPU.
            from sglang.srt.hardware_backend.npu.modules.minimax_m3_processor import (
                npu_apply_minimax_m3_image_preprocess_patch,
                npu_apply_minimax_m3_video_preprocess_patch,
            )

            npu_apply_minimax_m3_image_preprocess_patch(processor.image_processor)
            if (
                hasattr(processor, "video_processor")
                and processor.video_processor is not None
            ):
                npu_apply_minimax_m3_video_preprocess_patch(processor.video_processor)
            return "npu"
        if processor.__class__.__name__ not in {"Glm4vProcessor", "Glm46VProcessor"}:
            # For qwen-vl, the processor hits a reshape issue from the Ascend
            # dims restriction.
            from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
                npu_apply_qwen_image_preprocess_patch,
            )

            npu_apply_qwen_image_preprocess_patch()
            return "npu"
        if processor.__class__.__name__ == "Glm46VProcessor":
            from sglang.srt.hardware_backend.npu.modules.glm46v_processor import (
                npu_apply_glm46v_image_preprocess_patch,
            )

            npu_apply_glm46v_image_preprocess_patch()
            return "npu"
        return None

    @contextmanager
    def _temporary_fast_processor_cuda_pool(self, device: Optional[str]):
        """Release fast-processor CUDA temporaries after CPU feature transport."""
        can_release = (
            device is not None
            and torch.device(device).type == "cuda"
            and not self.keep_mm_features_on_device
            and not self.precompute_hash_before_cpu_transfer
        )
        if not can_release:
            yield
            return

        with torch.cuda.device(device):
            pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool, device=device):
            yield

    @classmethod
    def _load_single_item(
        cls,
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.

        If data is processor_output or precomputed embedding, return directly.

        Class method that can be pickled for multiprocessing
        """
        if cls._is_preprocessed_input(data):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data, cls.gpu_image_decode)
                if isinstance(img, torch.Tensor):
                    return img  # JPEG already decoded on GPU by nvJPEG
                # PIL decodes lazily; do it here in the io worker so the decode
                # doesn't run later on the event-loop thread.
                if discard_alpha_channel:
                    if cls.smart_rgb_conversion:
                        return smart_to_rgb(img)
                    if img.mode != "RGB":
                        return img.convert("RGB")
                img.load()
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except CLIENT_MEDIA_EXCEPTIONS as e:
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            raise ValueError(f"Error while loading data {data_str}: {e}") from e
        except Exception as e:
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            raise RuntimeError(f"Error while loading data {data_str}: {e}") from e

    @staticmethod
    def _get_preprocessed_input_format(data):
        """returns the detailed format if the provided data is already preprocessed.
        returns none if the provided data is not preprocessed
        """
        if not isinstance(data, dict):
            return None
        data_format = data.get("format")
        if isinstance(data_format, MultimodalInputFormat):
            return data_format
        if data_format in (
            MultimodalInputFormat.PROCESSOR_OUTPUT.name,
            "processor_output",
        ):
            return MultimodalInputFormat.PROCESSOR_OUTPUT
        if data_format in (
            MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
            "precomputed_embedding",
        ):
            return MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        return None

    @classmethod
    def _is_preprocessed_input(cls, data):
        """returns if the data is already preprocessed (by the vlm processor)"""
        return cls._get_preprocessed_input_format(data) is not None

    @classmethod
    def _all_mm_data_is_preprocessed(cls, *data_lists):
        has_mm_data = False
        for data_list in data_lists:
            if not data_list:
                continue
            if not isinstance(data_list, list):
                data_list = [data_list]
            for item in data_list:
                if item is None:
                    continue
                has_mm_data = True
                if not cls._is_preprocessed_input(item):
                    return False
        return has_mm_data

    def _submit_mm_data_loading_tasks_simple(
        self,
        data_list: Optional[list],
        modality: Modality,
        audio_sample_rate: Optional[int],
        discard_alpha_channel: bool,
    ) -> List[Tuple[Modality, int, concurrent.futures.Future]]:
        """
        Simple version: For one modal data submit IO load task.
        Return:
            List[(modality, index_in_that_modality, future)]
        """
        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []

        if not data_list:
            logger.debug(
                "[_submit_mm_data_loading_tasks_simple] no data for modality=%s",
                modality.name,
            )
            return futures

        for idx, data in enumerate(data_list):
            logger.debug(
                "[_submit_mm_data_loading_tasks_simple] submit load task: "
                "modality=%s, index=%d, data_type=%s",
                modality.name,
                idx,
                type(data),
            )
            item_sample_rate = audio_sample_rate
            if isinstance(data, dict) and "url" in data:
                if modality == Modality.AUDIO:
                    options = data.get("preprocess_kwargs") or {}
                    item_sample_rate = options.get("sampling_rate", audio_sample_rate)
                data = data["url"]
            future = self.io_executor.submit(
                self.__class__._load_single_item,
                data,
                modality,
                None,  # frame_count_limit: no consider for fast path
                item_sample_rate,
                discard_alpha_channel,
            )
            futures.append((modality, idx, future))

        return futures

    @staticmethod
    def _validate_one_modality(modality: Modality, data_list: Optional[list]):
        if data_list is None:
            return
        if not isinstance(data_list, list):
            raise TypeError(
                f"{modality.name} must be a list or None, got {type(data_list)}"
            )

        formatted_indices = []
        for idx, item in enumerate(data_list):
            if MultimodalProcessorMixin._is_preprocessed_input(item):
                formatted_indices.append(idx)

        if formatted_indices:
            if len(data_list) != 1:
                raise ValueError(
                    f"For {modality}, when providing a 'processor_output' or "
                    f"'precomputed_embedding', you must pass exactly one item; "
                    f"received {len(data_list)} items (formatted at indices {formatted_indices})."
                )

    @staticmethod
    def validate_mm_data(
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
    ):
        """
        Validate multimodal input lists per modality.

        Rule per modality (image/video/audio):
        - Either the list has exactly one item and that single item is a dict with
          format in {"processor_output", "precomputed_embedding"};
        - Or, the list contains only "normal" items (i.e., does not include any
          item whose format is one of the two above).

        Empty or None lists are considered valid.
        """

        MultimodalProcessorMixin._validate_one_modality(Modality.IMAGE, image_data)
        MultimodalProcessorMixin._validate_one_modality(Modality.VIDEO, video_data)
        MultimodalProcessorMixin._validate_one_modality(Modality.AUDIO, audio_data)

    async def fast_load_mm_data(
        self,
        prompt: Optional[Union[str, List[int]]],
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
        input_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        A fast version of `load_mm_data` that loads multimodal data directly.
        This version does not scan the prompt to recognize tokens. It assumes
        that the caller has already aligned the tokens and data in a 1:1 manner.
        The behavior is as follows:
          1. It runs `_load_single_item` for all input data concurrently.
          2. It returns the loaded images, videos, and audios in their original order.
          3. It returns text, or passes supported token IDs through unchanged.
        A missing prompt preserves per-source processing options with the media.
        """

        if self.use_token_space_processor and isinstance(prompt, list):
            if input_ids is None:
                input_ids = prompt
            prompt = None

        # Convert prompt into str
        if prompt is None:
            prompt_str = ""
        elif isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt_str = self._tokenizer.decode(prompt)
        else:
            assert isinstance(prompt, str)
            prompt_str = prompt

        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []

        modalities_data = [
            (image_data, Modality.IMAGE),
            (video_data, Modality.VIDEO),
            (audio_data, Modality.AUDIO),
        ]

        for data_list, modality in modalities_data:
            futures.extend(
                self._submit_mm_data_loading_tasks_simple(
                    data_list, modality, audio_sample_rate, discard_alpha_channel
                )
            )

        logger.debug("[load_mm_data(simple)] total futures submitted: %d", len(futures))

        images: List[Any] = [None] * len(image_data) if image_data else []
        videos: List[Any] = [None] * len(video_data) if video_data else []
        audios: List[Any] = [None] * len(audio_data) if audio_data else []

        if prompt is None:
            media_sources = {modality: data for data, modality in modalities_data}

        for modality, idx, future in futures:
            try:
                result = await asyncio.wrap_future(future)
            except ValueError as e:
                logger.info(
                    "[load_mm_data(simple)] invalid %s data at index=%d: %s",
                    modality.name,
                    idx,
                    e,
                )
                raise ValueError(
                    f"An exception occurred while loading {modality.name} data "
                    f"at index {idx}: {e}"
                ) from e
            except Exception as e:
                logger.exception(
                    "[load_mm_data(simple)] error loading %s data at index=%d",
                    modality.name,
                    idx,
                )
                raise RuntimeError(
                    f"An exception occurred while loading {modality.name} data at index {idx}: {e}"
                )

            if prompt is None:
                source = media_sources[modality][idx]
                if isinstance(source, (ImageData, VideoData)):
                    source = {
                        "url": source.url,
                        "preprocess_kwargs": source.preprocess_kwargs,
                    }
                if isinstance(source, dict) and "url" in source:
                    result = {**source, "url": result}

            if modality == Modality.IMAGE:
                images[idx] = result
            elif modality == Modality.VIDEO:
                videos[idx] = result
            elif modality == Modality.AUDIO:
                audios[idx] = result

        logger.debug(
            "[load_mm_data(simple)] loaded counts: images=%d, videos=%d, audios=%d",
            len(images),
            len(videos),
            len(audios),
        )

        return BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text=prompt_str,
            input_ids=input_ids,
        )
