"""Integration tests using respx to mock HTTP responses."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from shrimp_vision_router.backends import BackendManager
from shrimp_vision_router.models import BackendConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_completion_response(
    model: str = "some-model",
    content: str = "Hello",
    finish_reason: str = "stop",
) -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


# ---------------------------------------------------------------------------
# AC1: Health check returns all backends as keys
# ---------------------------------------------------------------------------


def test_health_returns_all_backends(client):
    """GET /health includes both backend names as keys."""
    with respx.mock(base_url="http://mini-1:8080") as mini1, \
         respx.mock(base_url="http://mini-2:8080") as mini2:
        mini1.head("/health").respond(200)
        mini2.head("/health").respond(200)
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "powerful-mini" in data
    assert "fast-mini" in data


# ---------------------------------------------------------------------------
# AC2: Chat completions proxies to the correct backend and returns 200
# ---------------------------------------------------------------------------


def test_chat_completions_proxies_correct_backend(client):
    """POST /v1/chat/completions with model='llama3.1' routes to powerful-mini."""
    with respx.mock(base_url="http://mini-1:8080/v1") as mini1:
        mini1.post("/chat/completions").respond(
            200, json=make_completion_response()
        )
        response = client.post(
            "/v1/chat/completions",
            json={"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    data = response.json()
    # model field should be replaced with the requested model name
    assert data["model"] == "llama3.1"
    assert data["choices"][0]["message"]["content"] == "Hello"


# ---------------------------------------------------------------------------
# AC3: Unknown model returns 400
# ---------------------------------------------------------------------------


def test_chat_completions_unknown_model_returns_400(client):
    """POST with model='nonexistent-model' returns 400."""
    response = client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 400
    assert "error" in response.json()
    assert "nonexistent-model" in response.json()["error"]


# ---------------------------------------------------------------------------
# AC4: Backend returning 503 results in 502
# ---------------------------------------------------------------------------


def test_chat_completions_backend_error_returns_502():
    """Backend returning 503 raises HTTPStatusError → app returns 502 Bad Gateway."""
    from unittest.mock import AsyncMock, MagicMock

    from fastapi.testclient import TestClient

    from shrimp_vision_router.app import create_app

    backends = {
        "powerful-mini": BackendConfig(
            base_url="http://mini-1:8080/v1",
            models=["llama3.1"],
            max_concurrency=1,
        ),
    }
    manager = BackendManager(backends, request_timeout_seconds=10)

    # Simulate a 503 response from the backend
    error_response = httpx.Response(
        503,
        content=b"",
        request=MagicMock(),
    )
    manager.forward_chat = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Backend unavailable",
            request=MagicMock(),
            response=error_response,
        )
    )

    app = create_app({
        "backends": {
            "powerful-mini": {
                "base_url": "http://mini-1:8080/v1",
                "models": ["llama3.1"],
                "max_concurrency": 1,
            }
        }
    })
    app.state.backend_manager = manager
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.1", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502
    assert response.json()["error"] == "Backend error"


# ---------------------------------------------------------------------------
# AC5: Concurrency limit enforced -- requests queued, not dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limit_enforced():
    """Three concurrent requests to same backend are queued by the semaphore.

    With max_concurrency=1 and a 1-second backend delay, all 3 requests
    must complete within 5 seconds (3 * 1s + buffer) without a timeout.
    """
    DELAY = 1.0  # seconds
    NUM_REQUESTS = 3
    TIMEOUT = 5.0  # must complete within this

    manager = BackendManager(
        backends={
            "limited": BackendConfig(
                base_url="http://slow:8080/v1",
                models=["llama3.1"],
                max_concurrency=1,  # only one at a time
            )
        },
        request_timeout_seconds=10,
    )

    start_times: list[float] = []
    end_times: list[float] = []

    async def slow_forward(name, backend, payload):
        import time
        start_times.append(time.monotonic())
        await asyncio.sleep(DELAY)
        end_times.append(time.monotonic())
        mock_resp = httpx.Response(
            200,
            json=make_completion_response(),
            request=httpx.Request("POST", "http://slow:8080/v1/chat/completions"),
        )
        return mock_resp

    manager.forward_chat = slow_forward

    from shrimp_vision_router.router import route_and_forward, make_chat_response

    async def single_request(model: str):
        from shrimp_vision_router.models import ChatCompletionRequest
        req = ChatCompletionRequest(model=model, messages=[{"role": "user", "content": "hi"}])
        result = await route_and_forward(manager, req)
        return await make_chat_response(result, model)

    async def run_all():
        return await asyncio.gather(
            *[single_request("llama3.1") for _ in range(NUM_REQUESTS)]
        )

    # Run with timeout -- should NOT raise TimeoutError
    result = await asyncio.wait_for(run_all(), timeout=TIMEOUT)

    assert result is not None
    assert len(result) == NUM_REQUESTS

    # Verify sequential execution: each start_time should be >= previous end_time
    for i in range(1, len(start_times)):
        assert start_times[i] >= end_times[i - 1], (
            f"Request {i} started before request {i-1} finished -- "
            f"semaphore queueing violated"
        )

    await manager.close()
