"""Placeholder contract for future loaded-media and processed-media caches."""

from typing import Any

from sglang.srt.multimodal.modality import Modality


class MediaCache:
    """Default to cache misses; storage and identity policies are deferred."""

    def get(self, modality: Modality, source: Any, options: dict) -> Any:
        """Return a request-owned media value, or None on a cache miss."""
        return None

    def put(self, modality: Modality, source: Any, options: dict, value: Any) -> None:
        """Accept one media value; the placeholder retains nothing."""
        pass
