from transformers import BatchFeature

from sglang.srt.multimodal.media_processing import process_media_groups
from sglang.srt.multimodal.media_processor import MultimodalProcessorMixin
from sglang.srt.multimodal.mm_token_expansion import expand_token_placeholders


class TokenSpaceMultimodalProcessor(MultimodalProcessorMixin):
    """Prepare native media fields and expand multimodal placeholders in token space."""

    supports_token_expansion = True
    use_token_space_processor = True
    media_processor_kwargs_type = None

    def __init__(self, hf_config, processor, **kwargs):
        self._initialize_processor(hf_config, _processor=processor, **kwargs)

    def process_media(
        self,
        *,
        images=None,
        videos=None,
        audios=None,
        processor=None,
        image_device=None,
        video_device=None,
        **kwargs,
    ) -> BatchFeature:
        """Process loaded media into native model fields and expansion metadata.

        Accept the modality lists returned by `fast_load_mm_data`; token expansion
        and serving assembly remain separate from this prompt-independent stage.
        """
        processor = self._processor if processor is None else processor
        processor_device = None
        if (images or videos) and not self.disable_fast_image_processor:
            processor_device = self._fast_image_processor_device(processor)
            if image_device is None:
                image_device = processor_device
        if videos and self.video_preprocessing_device is not None:
            image_device = video_device = self.video_preprocessing_device
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", True)
        processor_kwargs = processor._merge_kwargs(
            self.media_processor_kwargs_type or processor.valid_processor_kwargs,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
            **kwargs,
        )
        image_kwargs = processor_kwargs["images_kwargs"]
        video_kwargs = processor_kwargs["videos_kwargs"]
        audio_kwargs = processor_kwargs["audio_kwargs"]
        image_kwargs.update(self.image_config)
        video_kwargs.update(self.video_config)
        audio_kwargs.update(self.audio_config)
        if image_device is not None:
            image_kwargs["device"] = image_device
            if videos:
                video_kwargs.setdefault("device", image_device)
        if video_device is not None:
            video_kwargs["device"] = video_device
        with self._temporary_fast_processor_cuda_pool(processor_device):
            media_features = BatchFeature()
            for sources, process_modality, options in (
                (images, self.process_images, image_kwargs),
                (videos, self.process_videos, video_kwargs),
                (audios, self.process_audio, audio_kwargs),
            ):
                if not sources:
                    continue
                media_sources, source_configs = [], []
                for source in sources:
                    if isinstance(source, dict) and "url" in source:
                        media_sources.append(source["url"])
                        source_configs.append(source.get("preprocess_kwargs") or {})
                    else:
                        media_sources.append(source)
                        source_configs.append({})
                if any(source_configs):
                    options["source_configs"] = source_configs
                modality_features = process_modality(
                    media_sources, processor, **options
                )
                duplicate_keys = media_features.keys() & modality_features.keys()
                if duplicate_keys:
                    raise ValueError(
                        f"Conflicting encoder input fields: {sorted(duplicate_keys)}"
                    )
                media_features.update(modality_features)
        return media_features

    def process_images(self, images, processor, **kwargs):
        grouped = process_media_groups(images, processor, self.process_images, kwargs)
        if grouped is not None:
            return grouped
        return dict(processor.image_processor(images, **kwargs))

    def process_videos(self, videos, processor, **kwargs):
        raise NotImplementedError

    def process_audio(self, audios, processor, **kwargs):
        raise NotImplementedError

    def get_mm_token_expansion_spec(self, processor, media_features):
        raise NotImplementedError

    mm_token_expansion = staticmethod(expand_token_placeholders)
