"""Wall-clock marks for the rollout HTTP path.

The client reconstructs one waterfall per request across three processes, so it
needs absolute marks, not durations. They ride back on a response header: every
mark is taken before the response exists, and a header leaves the msgpack body
contract untouched for clients that ignore it.
"""

from __future__ import annotations

import json
import time

TIMING_HEADER = "x-sgld-timing"

# Abbreviated because the whole map travels in one HTTP header.
WIRE_KEYS = {
    "srv_recv": "rc",
    "forward_start": "fs",
    "forward_end": "fe",
    "build_start": "bs",
    "build_end": "be",
    "dump_end": "de",
    "msgpack_end": "me",
}


class RequestStamps:
    """Absolute wall-clock marks for one rollout request."""

    __slots__ = ("request_id", "_marks")

    def __init__(self, request_id: str = "") -> None:
        self.request_id = request_id
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        assert name in WIRE_KEYS, f"unknown timing mark {name!r}"
        self._marks[name] = time.time()

    def to_header(self) -> str:
        payload: dict[str, object] = {
            WIRE_KEYS[name]: round(t, 6) for name, t in self._marks.items()
        }
        if self.request_id:
            payload["rid"] = self.request_id
        return json.dumps(payload, separators=(",", ":"))
