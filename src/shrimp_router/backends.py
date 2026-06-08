"""Backend pool management, health checks, and request forwarding."""
from __future__ import annotations

import asyncio
import itertools
import math
import threading
import time
from collections import deque

import httpx

from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitOpenError
from .models import BackendConfig, RouterConfig
from .quota import QuotaRegistry


class BackendPool:
    """Weighted round-robin pool over a set of backends."""


    def __init__(self, names: list[str], backends: dict[str, BackendConfig]) -> None:
        # Expand by weight: a backend with weight=2 appears twice in rotation
        self._rotation: list[str] = []
        for name in names:
            cfg = backends.get(name)
            weight = cfg.weight if cfg else 1
            self._rotation.extend([name] * weight)
        self._cycle = itertools.cycle(self._rotation)
        self._lock = asyncio.Lock()

    async def next(self) -> str:
        async with self._lock:
            return next(self._cycle)



class BackendMetrics:
    """Thread-safe per-backend request counter and latency sampler."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._lock = threading.Lock()
        self._request_count: int = 0
        self._error_count: int = 0
        self._latencies: deque[float] = deque(maxlen=max_samples)

    def record_request(self, latency_seconds: float, is_error: bool) -> None:
        with self._lock:
            self._request_count += 1
            if is_error:
                self._error_count += 1
            self._latencies.append(latency_seconds)

    def stats(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies)
            n = len(latencies)
            p50 = latencies[math.floor(n * 0.50)] if n > 0 else 0.0
            p95 = latencies[math.floor(n * 0.95)] if n > 0 else 0.0
            return {
                "request_count": self._request_count,
                "error_count": self._error_count,
                "latency_p50_seconds": round(p50, 4),
                "latency_p95_seconds": round(p95, 4),
            }


class BackendManager:
    """Manages backends, concurrency, quota, health checks, and forwarding."""

    def __init__(self, config: RouterConfig) -> None:
        self.backends = config.backends
        self.routing = config.routing
        self._timeout = config.request_timeout_seconds
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._quota = QuotaRegistry()
        self._health: dict[str, float] = {}  # name -> last_healthy timestamp
        self._metrics: dict[str, BackendMetrics] = {
            name: BackendMetrics() for name in config.backends
        }


        # Build VLM pool
        vision_names = config.routing.vision_backends or [
            n for n, b in config.backends.items() if "vision" in b.tags or "vlm" in b.tags
        ]
        self._vlm_pool = BackendPool(vision_names, config.backends) if vision_names else None

        # Register quotas
        for name, backend in config.backends.items():
            if backend.quota:
                self._quota.register(name, backend.quota.window_seconds, backend.quota.max_requests)


        # Build circuit breaker registry (only for backends with config)
        self._cb_registry = CircuitBreakerRegistry()
        for name, backend in config.backends.items():
            if backend.circuit_breaker:
                cb_cfg = {
                    "failure_threshold": backend.circuit_breaker.failure_threshold,
                    "cooldown_seconds": backend.circuit_breaker.cooldown_seconds,
                }
                self._cb_registry.register(name, cb_cfg)

        # Per-backend health check intervals
        self._health_intervals: dict[str, float] = {
            name: (cfg.health_check_interval_seconds if cfg.health_check_interval_seconds else 30.0)
            for name, cfg in config.backends.items()
        }
        self._health_tasks: dict[str, asyncio.Task[None]] = {}
        self._health_next: dict[str, float] = {}  # name -> next poll timestamp
        self._health_lock = threading.Lock()

    def _semaphore(self, name: str) -> asyncio.Semaphore:
        if name not in self._semaphores:
            cfg = self.backends.get(name)
            limit = cfg.max_concurrency if cfg else 4
            self._semaphores[name] = asyncio.Semaphore(limit)
        return self._semaphores[name]

    def _auth_headers(self, name: str) -> dict[str, str]:
        cfg = self.backends.get(name)
        if not cfg:
            return {}
        key = cfg.api_key
        if key:
            return {"Authorization": f"Bearer {key}"}
        return {}

    def circuit_breaker(self, name: str) -> CircuitBreaker | None:
        """Return the circuit breaker for a backend, or None if not configured."""
        return self._cb_registry.get(name)


    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def record_request(self, name: str, latency_seconds: float, is_error: bool) -> None:
        """"Record a request outcome for metrics."""
        metrics = self._metrics.get(name)
        if metrics:
            metrics.record_request(latency_seconds, is_error)

    def metrics(self) -> dict:
        """Return live metrics for all backends."""
        result = {}
        for name in self.backends:
            m = self._metrics.get(name, BackendMetrics())
            cb = self.circuit_breaker(name)
            cb_state = cb.state.value if cb else "none"
            quota_stats = self._quota.stats().get(name, {})
            result[name] = {
                **m.stats(),
                "circuit_breaker_state": cb_state,
                "quota_used": quota_stats.get("used", 0),
                "quota_remaining": quota_stats.get("remaining", 0),
                "quota_rate_limited": quota_stats.get("rate_limited", False),
            }
        return result

    # ------------------------------------------------------------------
    # Health check scheduler
    # ------------------------------------------------------------------

    def start_health_checks(self) -> None:
        """Start background health check tasks for all backends."""
        for name in self.backends:
            task = asyncio.create_task(self._health_check_loop(name))
            self._health_tasks[name] = task

    async def _health_check_loop(self, name: str) -> None:
        """Continuously poll a single backend at its configured interval."""
        interval = self._health_intervals.get(name, 30.0)
        # Randomise first poll so backends don't all fire at once
        initial_delay = min(interval, 5.0)
        await asyncio.sleep(initial_delay)
        while True:
            await self.health_check(name)
            await asyncio.sleep(interval)

    async def health_check(self, name: str, timeout: float | None = None) -> bool:
        cfg = self.backends.get(name)
        if not cfg:
            return False
        # Resolve timeout: explicit arg > config field > 3.0 fallback
        resolved = timeout if timeout is not None else (
            cfg.health_check_timeout_seconds if cfg.health_check_timeout_seconds else 3.0
        )
        try:
            base = cfg.base_url.rstrip("/").removesuffix("/v1")
            path = cfg.health_check_path
            resp = await self._client.get(f"{base}{path}", timeout=httpx.Timeout(resolved))
            ok = 200 <= resp.status_code < 400
            if ok:
                self._health[name] = time.monotonic()
            return ok
        except Exception:
            return False

    async def forward(
        self,
        name: str,
        payload: dict,
        stream: bool = False,
        endpoint: str = "chat/completions",
    ) -> httpx.Response:
        cfg = self.backends[name]
        headers = {"Content-Type": "application/json", **self._auth_headers(name)}
        url = f"{cfg.base_url.rstrip('/')}/{endpoint}"

        start = time.monotonic()
        # Acquire semaphore only long enough to *send* the request, then release
        # immediately. This prevents slow upstream responses (e.g. Kimi K2.6
        # taking 5 minutes) from holding slots and starving the event loop.
        async with self._semaphore(name):
            self._quota.record(name)
            req = self._client.build_request("POST", url, json=payload, headers=headers)
            resp = await self._client.send(req, stream=True)
        # Semaphore released — slot is free for the next request

        if not stream:
            # Read the full body outside the semaphore so other requests aren't blocked
            await resp.aread()

        latency = time.monotonic() - start
        is_error = resp.status_code >= 400 or not (200 <= resp.status_code < 300)
        self.record_request(name, latency, is_error)

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 60))
            self._quota.record_429(name, retry_after)

        return resp

    async def pick_backends(
        self,
        task_type: str | None,
        is_vision: bool,
        phase: str | None = None,
    ) -> list[str]:
        """Return ordered list of backend names to try for this request."""
        if is_vision:
            # Vision: use round-robin VLM pool to distribute load, then fallback to rest
            vision = self.routing.vision_backends or [
                n for n, b in self.backends.items() if "vision" in b.tags or "vlm" in b.tags
            ]
            if not vision:
                return list(self.backends.keys())
            if self._vlm_pool:
                first = await self._vlm_pool.next()
                # Rotate: first = round-robin pick, rest = others in order as fallback
                rest = [n for n in vision if n != first]
                return [first] + rest
            return vision

        # Phase routing takes priority over task_type routing
        phase_backends = getattr(self.routing, "phase_backends", {}) or {}
        if phase and phase in phase_backends and phase_backends[phase]:
            return phase_backends[phase]

        # Text: check task_type routing next
        if task_type and task_type in self.routing.task_type_backends:
            ordered = self.routing.task_type_backends[task_type]
        elif self.routing.default_backends:
            ordered = self.routing.default_backends
        else:
            ordered = list(self.backends.keys())

        return ordered

    def next_available(self, candidates: list[str]) -> str | None:
        """Return first candidate with quota capacity, skipping rate-limited ones."""
        for name in candidates:
            if name in self.backends and self._quota.has_capacity(name):
                return name
        # All rate-limited -- return first anyway (let it 429 and we'll record it)
        return candidates[0] if candidates else None

    def quota_stats(self) -> dict:
        return self._quota.stats()

    def circuit_breaker_stats(self) -> dict:
        """Return circuit breaker status for all registered backends."""
        return self._cb_registry.status_all()

    async def close(self) -> None:
        for task in self._health_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()
