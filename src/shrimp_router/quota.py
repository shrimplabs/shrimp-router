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
        self._exhausted_until: float = 0.0  # provider-reported cutover, monotonic

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

    def set_exhausted_until(self, until: float) -> None:
        """Skip this backend until *until* (monotonic); 0 clears it."""
        with self._lock:
            self._exhausted_until = until

    @property
    def is_exhausted(self) -> bool:
        with self._lock:
            return time.monotonic() < self._exhausted_until

    def has_capacity(self) -> bool:
        if self.is_rate_limited or self.is_exhausted:
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
                "exhausted": t.is_exhausted,
            }
            for name, t in self._trackers.items()
        }


# ---------------------------------------------------------------------------
# Provider-specific live quota pollers
# ---------------------------------------------------------------------------

class MinimaxQuotaPoller:
    """Polls the MiniMax coding-plan quota API and, at *cutover_percent* used,
    marks every backend on that key exhausted until the window resets, so the
    router's fallbacks take over before MiniMax starts refusing.

    MiniMax endpoint:
      GET https://www.minimax.io/v1/api/openplatform/coding_plan/remains
    The "general" row's current_interval_remaining_percent is percent
    REMAINING; end_time (ms epoch) is when the window resets. (The old match
    on "MiniMax" in model_name never found that row, so this never fired.)
    """

    ENDPOINT = "https://www.minimax.io/v1/api/openplatform/coding_plan/remains"
    POLL_INTERVAL = 120  # seconds between polls

    def __init__(
        self,
        api_key: str,
        registry: "QuotaRegistry",
        backend_name: str = "minimax",
        poll_interval: int = POLL_INTERVAL,
        cutover_percent: float = 90.0,
        backend_names: list[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._registry = registry
        self._backend_name = backend_name
        self._backend_names = backend_names or [backend_name]
        self._poll_interval = poll_interval
        self._cutover = cutover_percent
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.used_percent: float | None = None

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
        rows = data.get("model_remains", [])
        row = next((r for r in rows if r.get("model_name") == "general"), None)
        if row is None or row.get("current_interval_remaining_percent") is None:
            return
        used = 100.0 - float(row["current_interval_remaining_percent"])
        self.used_percent = used
        until = 0.0
        if used >= self._cutover:
            try:
                left = float(row["end_time"]) / 1000.0 - time.time()
            except (KeyError, TypeError, ValueError):
                left = self._poll_interval
            # Re-checked every poll, so a missing/odd end_time costs one interval.
            until = time.monotonic() + max(left, self._poll_interval)
        for name in self._backend_names:
            tracker = self._registry.get(name)
            if tracker is None:
                continue
            was = tracker.is_exhausted
            tracker.set_exhausted_until(until)
            if until and not was:
                logger.warning(f"[quota-poller] {name}: MiniMax {used:.0f}% used >= {self._cutover:.0f}% "
                               f"-- skipping it until the window resets")
            elif was and not until:
                logger.info(f"[quota-poller] {name}: MiniMax {used:.0f}% used -- back in rotation")
        logger.info(f"[quota-poller] MiniMax {used:.1f}% used (cutover {self._cutover:.0f}%)")

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
