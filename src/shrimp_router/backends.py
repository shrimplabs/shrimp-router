"""Backend pool management, health checks, and request forwarding."""

from __future__ import annotations

import asyncio
import itertools
import time

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

        async with self._semaphore(name):
            self._quota.record(name)
            if stream:
                # Return the response without reading body -- caller streams it
                req = self._client.build_request("POST", url, json=payload, headers=headers)
                resp = await self._client.send(req, stream=True)
            else:
                resp = await self._client.post(url, json=payload, headers=headers)

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 60))
            self._quota.record_429(name, retry_after)

        return resp

    def quota_stats(self) -> dict:
        return self._quota.stats()

    def circuit_breaker_stats(self) -> dict:
        """Return circuit breaker status for all registered backends."""
        return self._cb_registry.status_all()

    async def close(self) -> None:
        await self._client.aclose()
