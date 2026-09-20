"""Group media sources with matching preprocessing settings."""

import torch


def process_media_groups(media, processor, process_modality, kwargs):
    """Process adjacent media with matching options and merge their feature fields.

    Args:
        media: Loaded sources of one modality, in their original order.
        processor: HF processor passed unchanged to process_modality.
        process_modality: Callable taking (sources, processor, **options) and returning feature fields.
        kwargs: Shared options; an aligned source_configs list is popped and overrides them per source.

    Returns:
        Feature fields merged in media order, or None when source_configs is absent or None.
    """
    source_configs = kwargs.pop("source_configs", None)
    if source_configs is None:
        return None
    groups = []
    for index, source in enumerate(media):
        options = {**kwargs, **source_configs[index]}
        if not groups or groups[-1][1] != options:
            groups.append(([], options))
        groups[-1][0].append(source)
    group_encoder_inputs = [
        process_modality(sources, processor, **options) for sources, options in groups
    ]
    if len(group_encoder_inputs) == 1:
        return group_encoder_inputs[0]
    encoder_inputs = {}
    for key in group_encoder_inputs[0]:
        group_values = [group_inputs[key] for group_inputs in group_encoder_inputs]
        if all(isinstance(value, torch.Tensor) for value in group_values) and all(
            value.shape[1:] == group_values[0].shape[1:] for value in group_values
        ):
            encoder_inputs[key] = torch.cat(group_values, dim=0)
        else:
            encoder_inputs[key] = [row for value in group_values for row in value]
    return encoder_inputs
