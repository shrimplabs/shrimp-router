"""Sliding-window quota tracking per backend, with live polling for MiniMax."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import httpx

logger = logging.getLogger(__name__)


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

    def update_max(self, name: str, new_max: int) -> None:
        """Update the quota ceiling for a backend (e.g. from a live API poll)."""
        tracker = self._trackers.get(name)
        if tracker:
            with tracker._lock:
                tracker._max = new_max

    def stats(self) -> dict[str, dict]:
        return {
            name: {
                "used": t.used,
                "remaining": t.remaining,
                "rate_limited": t.is_rate_limited,
            }
            for name, t in self._trackers.items()
        }


# ---------------------------------------------------------------------------
# Provider-specific live quota pollers
# ---------------------------------------------------------------------------

class MinimaxQuotaPoller:
    """Polls the MiniMax coding-plan quota API and updates the QuotaRegistry.

    MiniMax endpoint:
      GET https://www.minimax.io/v1/api/openplatform/coding_plan/remains
    Response fields we care about:
      current_interval_total_count   — total quota in this window
      current_interval_usage_count   — remaining (confusingly named)
    """

    ENDPOINT = "https://www.minimax.io/v1/api/openplatform/coding_plan/remains"
    POLL_INTERVAL = 300  # seconds between polls (5 minutes)

    def __init__(
        self,
        api_key: str,
        registry: "QuotaRegistry",
        backend_name: str = "minimax",
        poll_interval: int = POLL_INTERVAL,
    ) -> None:
        self._api_key = api_key
        self._registry = registry
        self._backend_name = backend_name
        self._poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _fetch(self) -> dict | None:
        try:
            resp = httpx.get(
                self.ENDPOINT,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"[quota-poller] MiniMax returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"[quota-poller] MiniMax poll failed: {e}")
        return None

    def _apply(self, data: dict) -> None:
        model_remains = data.get("model_remains", [])
        for entry in model_remains:
            name = entry.get("model_name", "")
            if "MiniMax" not in name:
                continue
            total = entry.get("current_interval_total_count", 0)
            remaining = entry.get("current_interval_usage_count", 0)  # remaining, despite name
            used = total - remaining
            if total > 0:
                # Set ceiling to 90% of total to cut over before hitting the wall
                ceiling = int(total * 0.90)
                self._registry.update_max(self._backend_name, ceiling)
                # Sync our used count: if the API says more is used than we've counted,
                # inject synthetic timestamps to bring our counter up to date
                tracker = self._registry.get(self._backend_name)
                if tracker and tracker.used < used:
                    gap = used - tracker.used
                    with tracker._lock:
                        now = time.monotonic()
                        for _ in range(gap):
                            tracker._timestamps.append(now)
                logger.info(
                    f"[quota-poller] {self._backend_name}: {used}/{total} used "
                    f"(ceiling set to {ceiling})"
                )
            break

    def poll_once(self) -> None:
        data = self._fetch()
        if data:
            self._apply(data)

    def start(self) -> None:
        """Start background polling thread."""
        self.poll_once()  # immediate first poll
        self._thread = threading.Thread(target=self._loop, daemon=True, name="minimax-quota-poller")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self._poll_interval):
            self.poll_once()
