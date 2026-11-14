"""Protection for the expensive endpoints of a public deployment (the optimiser and the viscous check).

Two limits, both answered with a clear message instead of a slow or failed request:
  * at most ``AERONET_MAX_HEAVY`` of these requests run at once (others get 503, "busy, try again in a moment");
  * at most ``AERONET_HEAVY_PER_MIN`` per client address per minute (429).
State lives in this process, which is enough for a single small server; behind several workers each has its own limits.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from . import config

_lock = threading.Lock()
_running = 0
_recent: dict[str, deque] = defaultdict(deque)


def reset() -> None:
    """Tests only."""
    global _running
    with _lock:
        _running = 0
        _recent.clear()


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def heavy_guard(request: Request):
    """FastAPI dependency (with cleanup): take a slot, or refuse."""
    global _running
    now = time.monotonic()
    who = _client(request)
    with _lock:
        window = _recent[who]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= config.heavy_per_minute():
            raise HTTPException(status_code=429, detail="Too many requests from this address. Please wait a minute and try again.")
        if _running >= config.max_heavy():
            raise HTTPException(status_code=503, detail="The server is busy with other requests. Please try again in a moment.")
        window.append(now)
        _running += 1
    try:
        yield
    finally:
        with _lock:
            _running -= 1
