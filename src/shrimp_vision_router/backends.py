"""Per-backend concurrency management and health checks."""

from __future__ import annotations

import asyncio
import httpx

from .models import BackendConfig


class PerBackendSemaphore:
    """Thin wrapper holding an asyncio.Semaphore per backend name."""

    def __init__(self) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self, name: str, max_concurrency: int | None = None
    ) -> asyncio.Semaphore:
        async with self._lock:
            if name not in self._semaphores:
                self._semaphores[name] = asyncio.Semaphore(max_concurrency or 1)
            return self._semaphores[name]

    async def release(self, name: str) -> None:
        async with self._lock:
            if name in self._semaphores:
                self._semaphores[name].release()


class BackendManager:
    """Manages backends, concurrency limits, and health checks."""

    def __init__(
        self,
        backends: dict[str, BackendConfig],
        request_timeout_seconds: int,
    ) -> None:
        self.backends = backends
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_seconds),
            limits=httpx.Limits(),
        )
        self._sem = PerBackendSemaphore()

    def backends_for_model(self, model: str) -> list[BackendConfig]:
        """Return all backends whose models list contains the requested model name."""
        return [b for b in self.backends.values() if model in b.models]

    async def health_check(
        self, name: str, backend: BackendConfig, timeout: float = 5.0
    ) -> bool:
        """Return True on 2xx/3xx, False on timeout or error."""
        try:
            base = backend.base_url.replace("/v1", "")
            path = backend.health_check_path or "/health"
            response = await self._client.head(
                f"{base}{path}", timeout=httpx.Timeout(timeout)
            )
            return 200 <= response.status_code < 400
        except Exception:
            return False

    async def forward_chat(
        self, name: str, backend: BackendConfig, request_payload: dict
    ) -> httpx.Response:
        """POST to the backend's /chat/completions endpoint.

        Must be called within an acquired semaphore context.
        """
        return await self._client.post(
            f"{backend.base_url}/chat/completions",
            json=request_payload,
        )

    async def acquire(self, name: str) -> asyncio.Semaphore:
        """Acquire the semaphore for a named backend."""
        backend = self.backends.get(name)
        max_concurrency = backend.max_concurrency if backend else None
        return await self._sem.acquire(name, max_concurrency)

    async def release(self, name: str) -> None:
        """Release the semaphore for a named backend."""
        await self._sem.release(name)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
