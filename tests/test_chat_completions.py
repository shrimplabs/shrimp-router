"""Tests for the /v1/chat/completions proxy endpoint."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from shrimp_vision_router.app import create_app
from shrimp_vision_router.models import BackendConfig, ChatCompletionRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manager():
    """A BackendManager with mocked backends and HTTP client."""
    manager = MagicMock()
    manager.backends = {
        "cuda": BackendConfig(
            base_url="http://cuda:8080/v1",
            models=["llama3.2-vision"],
            max_concurrency=1,
        )
    }
    manager.backends_for_model = lambda model: [
        cfg for cfg in manager.backends.values() if model in cfg.models
    ]
    manager.acquire = AsyncMock()
    manager.release = AsyncMock()
    return manager


@pytest.fixture
def app_with_manager(mock_manager):
    app = create_app({
        "backends": {
            "cuda": {
                "base_url": "http://cuda:8080/v1",
                "models": ["llama3.2-vision"],
                "max_concurrency": 1,
            }
        }
    })
    app.state.backend_manager = mock_manager
    return app


@pytest.fixture
def client(app_with_manager):
    return TestClient(app_with_manager)


# ---------------------------------------------------------------------------
# AC1: Unknown model returns 400 JSON error
# ---------------------------------------------------------------------------


def test_unknown_model_returns_400(client, mock_manager):
    mock_manager.backends_for_model = lambda model: []
    mock_manager.acquire = AsyncMock()
    mock_manager.release = AsyncMock()

    response = client.post(
        "/v1/chat/completions",
        json={"model": "unknown-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 400
    assert "error" in response.json()
    assert "unknown-model" in response.json()["error"]


# ---------------------------------------------------------------------------
# AC2: Valid model calls forward_chat on the correct backend
# ---------------------------------------------------------------------------


def test_valid_model_calls_forward_chat(client, mock_manager):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama3.2-vision",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.status_code = 200
    mock_manager.forward_chat = AsyncMock(return_value=mock_response)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.2-vision", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "llama3.2-vision"
    mock_manager.forward_chat.assert_called_once()
    call_args = mock_manager.forward_chat.call_args
    assert call_args[0][0] == "cuda"  # backend_name


# ---------------------------------------------------------------------------
# AC4: Backend HTTP errors result in 502 JSON response
# ---------------------------------------------------------------------------


def test_backend_error_returns_502(client, mock_manager):
    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_manager.forward_chat = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Backend unavailable",
            request=MagicMock(),
            response=mock_response,
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.2-vision", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 502
    data = response.json()
    assert data["error"] == "Backend error"
    assert "detail" in data


# ---------------------------------------------------------------------------
# AC3: Per-backend semaphore limits are respected
# ---------------------------------------------------------------------------



@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_requests():
    """Requests beyond max_concurrency are queued until the semaphore frees."""

    max_concurrency = 1
    concurrent_count = 0
    max_seen = 0
    release_event = asyncio.Event()
    allow_release = False

    async def slow_forward(name, backend, payload):
        nonlocal concurrent_count, max_seen
        concurrent_count += 1
        max_seen = max(max_seen, concurrent_count)
        # Wait until signalled before releasing
        await release_event.wait()
        concurrent_count -= 1
        return MagicMock(json=lambda: {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1234567890,
            "model": backend.models[0],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })


    # Build a manager with a real PerBackendSemaphore
    from shrimp_vision_router.backends import BackendManager

    manager = BackendManager(
        backends={
            "cuda": BackendConfig(
                base_url="http://cuda:8080/v1",
                models=["llama3.2-vision"],
                max_concurrency=max_concurrency,
            )
        },
        request_timeout_seconds=10,
    )
    manager.forward_chat = slow_forward

    async def run_requests():
        nonlocal allow_release
        results = await asyncio.gather(
            *[
                _call_route(manager, "llama3.2-vision")
                for _ in range(3)
            ],
            return_exceptions=True,
        )
        allow_release = True
        release_event.set()
        return results

    max_seen_during = 0

    async def tracker():
        nonlocal max_seen_during
        tasks = [
            asyncio.create_task(_call_route(manager, "llama3.2-vision"))
            for _ in range(3)
        ]
        # Let all tasks start and queue up
        await asyncio.sleep(0.05)
        max_seen_during = max_seen  # capture concurrency level while tasks are queued
        release_event.set()
        await asyncio.gather(*tasks)

    await tracker()
    # At most max_concurrency requests should have been in-flight simultaneously
    assert max_seen_during <= max_concurrency, f"Expected ≤{max_concurrency} concurrent, saw {max_seen_during}"
    await manager.close()


async def _call_route(manager, model):
    from shrimp_vision_router.router import route_and_forward, make_chat_response
    request = ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
    )
    result = await route_and_forward(manager, request)
    return await make_chat_response(result, model)
