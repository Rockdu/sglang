"""Lossless tensor ↔ base64 serialization for HTTP transport.

Uses safetensors for dtype-preserving, zero-copy-friendly encoding.
Falls back to torch save/load when safetensors is unavailable.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import torch


def tensor_to_base64(t: torch.Tensor) -> str:
    """Serialize a CPU tensor to a base64 string (via safetensors if available)."""
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
    """Deserialize a base64 string back to a torch.Tensor."""
    raw = base64.b64decode(s)
    try:
        from safetensors.torch import load
        return load(raw)["t"]
    except ImportError:
        buf = io.BytesIO(raw)
        return torch.load(buf, weights_only=True)


def _maybe_serialize(obj: Any) -> Any:
    """Recursively convert tensors to base64 within nested dicts/lists."""
    if isinstance(obj, torch.Tensor):
        return {"__tensor__": True, "data": tensor_to_base64(obj), "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {k: _maybe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_serialize(v) for v in obj]
    return obj


def _maybe_deserialize(obj: Any) -> Any:
    """Inverse of ``_maybe_serialize``: restore tensors from ``__tensor__`` dicts."""
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            return base64_to_tensor(obj["data"])
        return {k: _maybe_deserialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_deserialize(v) for v in obj]
    return obj
