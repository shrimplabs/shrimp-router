"""Sliding-window quota tracking per backend."""

from __future__ import annotations

import threading
import time
from collections import deque


class QuotaTracker:
    """Thread-safe sliding-window request counter per backend."""

    def __init__(self, window_seconds: int, max_requests: int) -> None:
        self._window = window_seconds
        self._max = max_requests
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._rate_limited_until: float = 0.0

    def record(self) -> None:
        with self._lock:
            self._timestamps.append(time.monotonic())

    def record_429(self, retry_after: float = 60.0) -> None:
        """Mark backend as rate-limited for retry_after seconds."""
        with self._lock:
            self._rate_limited_until = time.monotonic() + retry_after

    @property
    def is_rate_limited(self) -> bool:
        with self._lock:
            return time.monotonic() < self._rate_limited_until

    @property
    def remaining(self) -> int:
        with self._lock:
            cutoff = time.monotonic() - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            return max(0, self._max - len(self._timestamps))

    @property
    def used(self) -> int:
        with self._lock:
            cutoff = time.monotonic() - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            return len(self._timestamps)

    def has_capacity(self) -> bool:
        if self.is_rate_limited:
            return False
        return self.remaining > 0


class QuotaRegistry:
    """Holds one QuotaTracker per backend name."""

    def __init__(self) -> None:
        self._trackers: dict[str, QuotaTracker] = {}

    def register(self, name: str, window_seconds: int, max_requests: int) -> None:
        self._trackers[name] = QuotaTracker(window_seconds, max_requests)

    def get(self, name: str) -> QuotaTracker | None:
        return self._trackers.get(name)

    def has_capacity(self, name: str) -> bool:
        tracker = self._trackers.get(name)
        if tracker is None:
            return True  # no quota configured → unlimited
        return tracker.has_capacity()

    def record(self, name: str) -> None:
        tracker = self._trackers.get(name)
        if tracker:
            tracker.record()

    def record_429(self, name: str, retry_after: float = 60.0) -> None:
        tracker = self._trackers.get(name)
        if tracker:
            tracker.record_429(retry_after)

    def stats(self) -> dict[str, dict]:
        return {
            name: {
                "used": t.used,
                "remaining": t.remaining,
                "rate_limited": t.is_rate_limited,
            }
            for name, t in self._trackers.items()
        }
