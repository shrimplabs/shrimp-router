"""Per-backend circuit breaker with closed/open/half-open states."""

from __future__ import annotations

import asyncio
import enum
import threading
import time
from dataclasses import dataclass, field


class CircuitState(str, enum.Enum):
    """Possible states for a circuit breaker."""

    CLOSED = "closed"      # normal operation
    OPEN = "open"          # failing, no requests allowed
    HALF_OPEN = "half_open"  # cooldown elapsed, one probe allowed


class CircuitOpenError(Exception):
    """Raised when a request hits an open circuit."""

    def __init__(self, backend: str, retry_after: float) -> None:
        self.backend = backend
        self.retry_after = retry_after
        super().__init__(f"Circuit is open for {backend}; retry after {retry_after:.0f}s")


@dataclass
class CircuitBreaker:
    """Thread-safe circuit breaker for a single backend.

    State machine:
      CLOSED    -- normal; requests pass through; on failure increment counter
                  when counter >= threshold, transition to OPEN
      OPEN      -- requests rejected; after cooldown_seconds, transition to HALF_OPEN
      HALF_OPEN -- one probe request allowed; on success -> CLOSED, on failure -> OPEN
    """

    name: str
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    _state: CircuitState = field(default=CircuitState.CLOSED)
    _failure_count: int = field(default=0)
    _opened_at: float = field(default=0.0)  # monotonic timestamp when we opened
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    @property
    def opened_at(self) -> float:
        with self._lock:
            return self._opened_at

    def should_allow_request(self) -> bool:
        """Synchronously check if a request may proceed.

        CLOSED   -> True
        OPEN     -> True only if cooldown has elapsed (transitions to HALF_OPEN)
        HALF_OPEN -> True (one probe request allowed)
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    return True
                return False
            # HALF_OPEN: allow the single probe
            return True

    def record_failure(self, is_timeout: bool = False) -> None:
        """Record a failed request (thread-safe, synchronous).

        CLOSED   -> increment counter; open if threshold reached
        HALF_OPEN -> reopen immediately
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._failure_count = 0
                return
            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()

    def record_success(self) -> None:
        """Record a successful request (thread-safe, synchronous).

        CLOSED   -> reset failure counter
        HALF_OPEN -> transition to CLOSED
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def can_request(self) -> bool:
        """Async: return True if allowed, raise CircuitOpenError if circuit is open.

        CLOSED   -> True
        OPEN     -> raise CircuitOpenError with remaining cooldown
        HALF_OPEN -> True
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN:
                return True
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    return True
                remaining = self.cooldown_seconds - elapsed
                raise CircuitOpenError(self.name, remaining)
        # Circuit was OPEN; sleep outside lock then re-check
        with self._lock:
            elapsed = time.monotonic() - self._opened_at
            remaining = max(0.0, self.cooldown_seconds - elapsed)
        await asyncio.sleep(remaining)
        with self._lock:
            if self._state == CircuitState.OPEN:
                self._state = CircuitState.HALF_OPEN
            return True

    def is_open(self) -> bool:
        """Return True only when state is OPEN (not HALF_OPEN)."""
        with self._lock:
            return self._state == CircuitState.OPEN

    def status(self) -> dict:
        """Return serialisable status dict for /health endpoint."""
        with self._lock:
            remaining_cooldown = 0.0
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                remaining_cooldown = max(0.0, self.cooldown_seconds - elapsed)
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "remaining_cooldown_seconds": round(remaining_cooldown, 2),
            }



class CircuitBreakerRegistry:
    """Holds one CircuitBreaker per backend name."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def register(self, name: str, config: dict | None = None) -> CircuitBreaker:
        """Create (or return existing) breaker for a backend."""
        with self._lock:
            if name in self._breakers:
                return self._breakers[name]
            cb = CircuitBreaker(
                name=name,
                failure_threshold=(config or {}).get("failure_threshold", 3),
                cooldown_seconds=float((config or {}).get("cooldown_seconds", 60)),
            )
            self._breakers[name] = cb
            return cb

    def get(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def stats(self) -> dict[str, dict]:
        return {name: cb.status() for name, cb in self._breakers.items()}

    def status_all(self) -> dict[str, dict]:
        return self.stats()
