"""Tensor serialization for post-training / rollout HTTP.

Encodes ``torch.Tensor`` into JSON-friendly blobs (base64 + metadata) for httpx / ORJSON. Use
``_maybe_deserialize`` on the client or in tests to restore tensors. Prefers ``safetensors`` when available
(dtype-friendly); falls back to ``torch.save`` / ``torch.load`` otherwise.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import torch


def tensor_to_base64(t: torch.Tensor) -> str:
    """Encode one tensor as an ASCII base64 string (always CPU, contiguous)."""
    t = t.detach().contiguous().cpu()
    try:
        from safetensors.torch import save

        raw = save({"t": t})
    except ImportError:
        buf = io.BytesIO()
        torch.save(t, buf)
        raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii")


def base64_to_tensor(s: str) -> torch.Tensor:
    """Inverse of ``tensor_to_base64``."""
    raw = base64.b64decode(s)
    try:
        from safetensors.torch import load

        return load(raw)["t"]
    except ImportError:
        buf = io.BytesIO(raw)
        return torch.load(buf, weights_only=True)


def _maybe_serialize(obj: Any) -> Any:
    """Walk dict/list/tuple recursively; replace each ``Tensor`` with ``{"__tensor__": True, "data": ..., "shape", "dtype"}``."""
    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "data": tensor_to_base64(obj),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, dict):
        return {k: _maybe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_serialize(v) for v in obj]
    return obj


def _maybe_deserialize(obj: Any) -> Any:
    """Inverse of ``_maybe_serialize``: detect dicts with ``__tensor__`` and run ``base64_to_tensor``; recurse otherwise."""
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            return base64_to_tensor(obj["data"])
        return {k: _maybe_deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_deserialize(v) for v in obj]
    return obj
