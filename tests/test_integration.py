"""Integration tests using respx to mock HTTP responses."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from shrimp_router.app import create_app


def _config():
    return {
        "backends": {
            "powerful-mini": {
                "base_url": "http://mini-1:8080/v1",
                "models": ["llama3.1"],
                "max_concurrency": 2, "response_format": "openai",
            },
            "fast-mini": {
                "base_url": "http://mini-2:8080/v1",
                "models": ["llama3.2"],
                "max_concurrency": 2, "response_format": "openai",
            },
        },
        "routing": {
            "default_backends": ["powerful-mini", "fast-mini"],
        },
    }


def _completion(content: str = "Hello") -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "llama3.1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_returns_all_backends():
    """GET /health includes both backend names."""
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    with respx.mock(assert_all_called=False) as mock:
        mock.head("http://mini-1:8080/health").respond(200)
        mock.head("http://mini-2:8080/health").respond(200)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "backends" in data
    assert "powerful-mini" in data["backends"]
    assert "fast-mini" in data["backends"]


# ---------------------------------------------------------------------------
# Chat completions proxies to backend and returns content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completions_proxies_correct_backend():
    """POST /v1/chat/completions routes to first backend and returns content."""
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://mini-1:8080/v1/chat/completions").respond(200, json=_completion("Hello"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello"


# ---------------------------------------------------------------------------
# All backends failing → 502
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completions_backend_error_returns_502():
    """All backends returning 503 → 502 Bad Gateway."""
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://mini-1:8080/v1/chat/completions").respond(503)
        mock.post("http://mini-2:8080/v1/chat/completions").respond(503)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert response.status_code == 502
    assert "error" in response.json()


# ---------------------------------------------------------------------------
# 429 on first backend falls back to second
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completions_429_falls_back():
    """First backend 429 → falls back to second backend."""
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://mini-1:8080/v1/chat/completions").respond(429)
        mock.post("http://mini-2:8080/v1/chat/completions").respond(200, json=_completion("Fallback"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "llama3.2", "messages": [{"role": "user", "content": "hi"}]},
            )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Fallback"


# ---------------------------------------------------------------------------
# Concurrency semaphore — all requests complete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_limit_no_requests_dropped():
    """Multiple concurrent requests all complete without being dropped."""
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://mini-1:8080/v1/chat/completions").respond(200, json=_completion())
        mock.post("http://mini-2:8080/v1/chat/completions").respond(200, json=_completion())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            results = await asyncio.gather(*[
                client.post("/v1/chat/completions", json={
                    "model": "llama3.1",
                    "messages": [{"role": "user", "content": "hi"}],
                })
                for _ in range(5)
            ])

    assert all(r.status_code == 200 for r in results)
