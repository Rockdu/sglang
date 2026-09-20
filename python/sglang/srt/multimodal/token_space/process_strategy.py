from transformers import BatchFeature

from sglang.srt.multimodal.mm_token_expansion import expand_token_placeholders
from sglang.srt.multimodal.modality import Modality


def bucket_by_config(configs):
    """Group source indices with equal configs, in order of first appearance."""
    config_buckets = []
    for index, config in enumerate(configs):
        for bucket_config, indices in config_buckets:
            if bucket_config == config:
                indices.append(index)
                break
        else:
            config_buckets.append((config, [index]))
    return config_buckets


class TokenSpaceProcessStrategy:
    """Prepare native media fields and expand multimodal placeholders in token space."""

    supports_token_expansion = True
    media_processor_kwargs_type = None

    def __init__(self, hf_config, processor, *, mm_process_config=None):
        self.hf_config = hf_config
        self._processor = processor
        mm_process_config = mm_process_config or {}
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})

    def process_media(
        self,
        *,
        images=None,
        videos=None,
        audios=None,
        image_source_configs=None,
        video_source_configs=None,
        audio_source_configs=None,
        processor=None,
        image_device=None,
        video_device=None,
        **kwargs,
    ) -> BatchFeature:
        """Process loaded media into native model fields and expansion metadata.

        Accept the modality lists returned by `fast_load_mm_data`; token expansion
        and serving assembly remain separate from this prompt-independent stage.
        Each optional source-config list is aligned with its loaded modality list.
        """
        processor = self._processor if processor is None else processor
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
        media_features = BatchFeature()
        for loaded_media, source_configs, process_modality, options in (
            (images, image_source_configs, self.process_images, image_kwargs),
            (videos, video_source_configs, self.process_videos, video_kwargs),
            (audios, audio_source_configs, self.process_audio, audio_kwargs),
        ):
            if source_configs is not None and len(source_configs) != len(
                loaded_media or []
            ):
                raise ValueError("Source configs must align with the loaded media list")
            if not loaded_media:
                continue
            if source_configs and any(source_configs):
                options["source_configs"] = source_configs
            modality_features = process_modality(loaded_media, processor, **options)
            duplicate_keys = media_features.keys() & modality_features.keys()
            if duplicate_keys:
                raise ValueError(
                    f"Conflicting encoder input fields: {sorted(duplicate_keys)}"
                )
            media_features.update(modality_features)
        return media_features

    def process_media_by_config_buckets(
        self,
        loaded_media,
        source_configs,
        processor,
        process_modality,
        kwargs,
        *,
        modality,
    ):
        """Process media in buckets of equal effective config and restore source order.

        Args:
            loaded_media: Loaded media of one modality, in their original source order.
            source_configs: Per-source options aligned with loaded_media; each overrides kwargs.
            processor: HF processor passed unchanged to process_modality.
            process_modality: Callable processing one source batch with shared options.
            kwargs: Options shared by every source.
            modality: Modality used by the model's feature splitting and assembly hooks.

        Returns:
            One modality's native feature fields in source order.
        """
        config_buckets = bucket_by_config(
            [{**kwargs, **source_config} for source_config in source_configs]
        )
        bucket_features = [
            process_modality(
                [loaded_media[index] for index in indices], processor, **config
            )
            for config, indices in config_buckets
        ]
        return self._restore_source_order(
            modality=modality,
            config_buckets=config_buckets,
            bucket_features=bucket_features,
        )

    def _restore_source_order(self, *, modality, config_buckets, bucket_features):
        """Merge per-bucket features into one output ordered like the original sources."""
        if len(bucket_features) == 1:
            return bucket_features[0]
        source_features = [None] * sum(len(indices) for _, indices in config_buckets)
        for (_, indices), features in zip(config_buckets, bucket_features):
            for index, source_feature in zip(
                indices, self.split_media_features(modality, features)
            ):
                source_features[index] = source_feature
        return self.merge_media_features(modality, source_features)

    def process_images(self, images, processor, *, source_configs=None, **kwargs):
        if source_configs is not None:
            return self.process_media_by_config_buckets(
                images,
                source_configs,
                processor,
                self.process_images,
                kwargs,
                modality=Modality.IMAGE,
            )
        return dict(processor.image_processor(images, **kwargs))

    def process_videos(self, videos, processor, **kwargs):
        raise NotImplementedError

    def process_audio(self, audios, processor, **kwargs):
        raise NotImplementedError

    def split_media_features(self, modality: Modality, features: dict) -> list[dict]:
        """Split one modality's native fields into ordered per-source fields."""
        raise NotImplementedError

    def merge_media_features(self, modality: Modality, features: list[dict]) -> dict:
        """Assemble ordered per-source fields using the model's batching rules."""
        raise NotImplementedError

    def get_mm_token_expansion_spec(self, processor, media_features):
        raise NotImplementedError

    mm_token_expansion = staticmethod(expand_token_placeholders)
