"""Media-only results shared by training and serving."""

from dataclasses import dataclass, field
from typing import Any

MediaId = tuple[str, int]


@dataclass
class ProcessedMediaItem:
    media_id: MediaId
    encoder_inputs: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    effective_options: dict[str, Any] = field(default_factory=dict)
    feature_name: str | None = None


@dataclass
class MediaProcessOutput:
    encoder_inputs: dict[str, Any]
    items: list[ProcessedMediaItem]


def process_media_groups(media, processor, process_modality, kwargs):
    """Group frozen options; each item's tensor fields partition batch axis zero."""
    import torch

    recipes = kwargs.pop("process_options", None)
    source_configs = kwargs.pop("source_configs", None)
    if recipes is None and source_configs is None:
        return None
    groups = []
    for index, source in enumerate(media):
        recipe = recipes[index] if recipes is not None else None
        options = (
            dict(recipe)
            if recipe is not None
            else {
                **kwargs,
                **(source_configs[index] if source_configs is not None else {}),
            }
        )
        if not groups or groups[-1][2] != options:
            groups.append(([], [], options))
        groups[-1][0].append(source)
        groups[-1][1].append(index)
    group_encoder_inputs = []
    processed_items = []
    for sources, indices, options in groups:
        processed_group = process_modality(sources, processor, **options)
        group_encoder_inputs.append(processed_group.encoder_inputs)
        for index, item in zip(indices, processed_group.items):
            item.media_id = (item.media_id[0], index)
            processed_items.append(item)
    if len(group_encoder_inputs) == 1:
        return MediaProcessOutput(
            encoder_inputs=group_encoder_inputs[0], items=processed_items
        )
    encoder_inputs = {}
    for key in group_encoder_inputs[0]:
        group_values = [group_inputs[key] for group_inputs in group_encoder_inputs]
        if all(isinstance(value, torch.Tensor) for value in group_values) and all(
            value.shape[1:] == group_values[0].shape[1:] for value in group_values
        ):
            encoder_inputs[key] = torch.cat(group_values, dim=0)
            offset = 0
            for item in processed_items:
                if item.encoder_inputs is None:
                    continue
                end = offset + item.encoder_inputs[key].shape[0]
                item.encoder_inputs[key] = encoder_inputs[key][offset:end]
                offset = end
        else:
            encoder_inputs[key] = [row for value in group_values for row in value]
    return MediaProcessOutput(encoder_inputs=encoder_inputs, items=processed_items)
