"""Tests for the packaged /v1/chat/completions router."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from shrimp_router.app import create_app
from shrimp_router.backends import BackendManager
from shrimp_router.models import RouterConfig


def _config(max_concurrency: int = 2) -> dict:
    return {
        "backends": {
            "primary": {
                "base_url": "http://primary:8080/v1",
                "models": ["primary-model"],
                "max_concurrency": max_concurrency,
                "format": "openai",
                "response_format": "openai",
            },
            "fallback": {
                "base_url": "http://fallback:8080/v1",
                "models": ["fallback-model"],
                "max_concurrency": max_concurrency,
                "format": "openai",
                "response_format": "openai",
            },
        },
        "routing": {
            "default_backends": ["primary", "fallback"],
        },
    }


def _completion(content: str = "Hello") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "primary-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.asyncio
async def test_valid_request_calls_first_configured_backend():
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        primary = mock.post("http://primary:8080/v1/chat/completions").respond(
            200,
            json=_completion("Primary"),
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "requested-model", "messages": [{"role": "user", "content": "hello"}]},
            )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Primary"
    assert response.json()["model"] == "requested-model"
    assert primary.called


@pytest.mark.asyncio
async def test_unknown_model_is_still_routed_by_policy():
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://primary:8080/v1/chat/completions").respond(200, json=_completion())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "unknown-model", "messages": [{"role": "user", "content": "hello"}]},
            )

    assert response.status_code == 200
    assert response.json()["model"] == "unknown-model"


@pytest.mark.asyncio
async def test_backend_http_errors_fall_back_then_return_502():
    app = create_app(_config())
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://primary:8080/v1/chat/completions").respond(503)
        mock.post("http://fallback:8080/v1/chat/completions").respond(503)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "requested-model", "messages": [{"role": "user", "content": "hello"}]},
            )

    assert response.status_code == 502
    assert "error" in response.json()


@pytest.mark.asyncio
async def test_per_backend_semaphore_limits_are_respected():
    manager = BackendManager(RouterConfig.model_validate(_config(max_concurrency=1)))
    concurrent_count = 0
    max_seen = 0
    release_event = asyncio.Event()

    async def slow_post(request: httpx.Request) -> httpx.Response:
        nonlocal concurrent_count, max_seen
        concurrent_count += 1
        max_seen = max(max_seen, concurrent_count)
        await release_event.wait()
        concurrent_count -= 1
        return httpx.Response(200, json=_completion(), request=request)

    transport = httpx.MockTransport(slow_post)
    await manager._client.aclose()
    manager._client = httpx.AsyncClient(transport=transport)

    try:
        tasks = [
            asyncio.create_task(manager.forward("primary", {"model": "requested-model", "messages": []}))
            for _ in range(3)
        ]
        await asyncio.sleep(0.05)
        assert max_seen <= 1
        release_event.set()
        responses = await asyncio.gather(*tasks)
    finally:
        await manager.close()

    assert all(response.status_code == 200 for response in responses)
