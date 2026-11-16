"""Tests for adaptive routing escalate-and-retry in router.handle_chat().

The adaptive routing upgrade lets shrimp-router offload ~95% of read-only
agent tokens to a cheap local backend (e.g. Athena) and only forward to a
strong/billed backend when the response actually contains a strong-tool
call. These tests exercise the escalate-and-retry path through the full
``/v1/chat/completions`` HTTP stack using respx to mock upstream backends.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from shrimp_router.app import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adaptive_config() -> dict:
    """Two-backend config with adaptive cheap/strong lists configured."""
    return {
        "backends": {
            "athena": {
                "base_url": "http://athena:8083/v1",
                "models": ["qwen3.6-35b-a3b-fast-256k"],
                "max_concurrency": 4,
                "format": "openai",
                "response_format": "openai",
            },
            "minimax": {
                "base_url": "http://minimax:8090/v1",
                "models": ["minimax-m3"],
                "max_concurrency": 4,
                "format": "openai",
                "response_format": "openai",
            },
        },
        "routing": {
            "default_backends": ["athena", "minimax"],
            "adaptive": True,
            "adaptive_cheap_backends": ["athena"],
            "adaptive_strong_backends": ["minimax"],
        },
    }


def _completion(content: str = "Hello") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen3.6-35b-a3b-fast-256k",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _messages_with_last_assistant_tool(tool_name: str) -> list[dict]:
    """Build a messages list whose last assistant turn has a tool_use block.

    Uses Anthropic-style ``content`` blocks (tool_use) because the strict
    ``ChatMessage`` Pydantic model strips OpenAI ``tool_calls``. The router
    reads the raw body and the tool_classifier handles both shapes.
    """
    return [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": tool_name,
                    "input": {},
                }
            ],
        },
    ]


def _text_only_messages() -> list[dict]:
    return [
        {"role": "user", "content": "hi"},
    ]


# ---------------------------------------------------------------------------
# Tool classification feeds pick_backends()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_classification_routes_to_cheap_backend():
    """When the last assistant turn was a read-only tool, pick athena (cheap)."""
    app = create_app(_adaptive_config())
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        athena = mock.post("http://athena:8083/v1/chat/completions").respond(
            200, json=_completion("Here is the file contents...")
        )
        mock.post("http://minimax:8090/v1/chat/completions").respond(
            200, json=_completion("STRONG")
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "messages": _messages_with_last_assistant_tool("read_file"),
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Here is the file contents..."
    assert athena.called


# ---------------------------------------------------------------------------
# Escalate-and-retry: cheap response with a strong tool
# ---------------------------------------------------------------------------


def _strong_tool_response(tool_name: str) -> dict:
    """OpenAI completion whose assistant message includes a strong tool_use."""
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen3.6-35b-a3b-fast-256k",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "doing it"},
                        {
                            "type": "tool_use",
                            "name": tool_name,
                            "input": {},
                        },
                    ],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }


@pytest.mark.asyncio
async def test_cheap_response_with_strong_tool_triggers_escalation():
    """Athena answers with a write_file tool call → escalates to minimax.

    The athena response MUST NOT reach the client.
    """
    app = create_app(_adaptive_config())
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        # Cheap backend returns a strong-tool response.
        athena_route = mock.post("http://athena:8083/v1/chat/completions").mock(
            side_effect=lambda req: httpx.Response(200, json=_strong_tool_response("write_file"))
        )
        # Strong backend returns a normal text completion.
        minimax_route = mock.post("http://minimax:8090/v1/chat/completions").mock(
            side_effect=lambda req: httpx.Response(200, json=_completion("From strong backend"))
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "messages": _messages_with_last_assistant_tool("read_file"),
                },
            )

    assert response.status_code == 200
    body = response.json()
    # Client must see the STRONG backend's text, not the athena tool_calls.
    assert body["choices"][0]["message"]["content"] == "From strong backend"
    assert body["_backend"] == "minimax"
    assert athena_route.called
    assert minimax_route.called
    # Athena response must be discarded -- no tool_calls in the returned message.
    msg = body["choices"][0]["message"]
    if isinstance(msg.get("content"), list):
        assert not any(
            b.get("type") == "tool_use" and b.get("name") in {"write_file", "edit_file"}
            for b in msg["content"]
        )


@pytest.mark.asyncio
async def test_cheap_response_with_read_tool_is_not_escalated():
    """Athena answering with read_file (a read tool) does NOT trigger escalation."""
    app = create_app(_adaptive_config())
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        athena = mock.post("http://athena:8083/v1/chat/completions").respond(
            200, json=_strong_tool_response("read_file")
        )
        mock.post("http://minimax:8090/v1/chat/completions").respond(200, json=_completion("X"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "messages": _messages_with_last_assistant_tool("read_file"),
                },
            )

    assert response.status_code == 200
    assert response.json()["_backend"] == "athena"
    assert athena.called


@pytest.mark.asyncio
async def test_unknown_tool_in_cheap_response_does_not_escalate():
    """An unrecognised tool in the cheap response is treated as not-strong."""
    app = create_app(_adaptive_config())
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        athena = mock.post("http://athena:8083/v1/chat/completions").respond(
            200, json=_strong_tool_response("weird_unknown_tool")
        )
        mock.post("http://minimax:8090/v1/chat/completions").respond(200, json=_completion("X"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "messages": _messages_with_last_assistant_tool("read_file"),
                },
            )

    assert response.status_code == 200
    assert response.json()["_backend"] == "athena"
    assert athena.called


# ---------------------------------------------------------------------------
# Streaming: skip escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_request_does_not_escalate():
    """stream=true responses skip escalate-and-retry and return directly."""
    app = create_app(_adaptive_config())
    transport = httpx.ASGITransport(app=app)

    sse_body = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"

    with respx.mock(assert_all_called=False) as mock:
        athena = mock.post("http://athena:8083/v1/chat/completions").respond(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )
        mock.post("http://minimax:8090/v1/chat/completions").respond(200, content=sse_body)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "stream": True,
                    "messages": _messages_with_last_assistant_tool("read_file"),
                },
            )

    assert response.status_code == 200
    assert athena.called
    assert b"data: [DONE]" in response.content


# ---------------------------------------------------------------------------
# Escalation cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_caps_at_two_hops():
    """Even if every strong backend also returns strong-tool responses, we cap at 2."""
    config = _adaptive_config()
    # Add a second strong backend so we have somewhere to escalate to.
    config["backends"]["opencode"] = {
        "base_url": "http://opencode:8787/v1",
        "models": ["kimi-k2.6"],
        "max_concurrency": 4,
        "format": "openai",
        "response_format": "openai",
    }
    config["routing"]["adaptive_strong_backends"] = ["minimax", "opencode"]

    app = create_app(config)
    transport = httpx.ASGITransport(app=app)

    call_count = {"athena": 0, "minimax": 0, "opencode": 0}

    def _route(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "athena" in url:
            call_count["athena"] += 1
            return httpx.Response(200, json=_strong_tool_response("write_file"))
        if "minimax" in url:
            call_count["minimax"] += 1
            return httpx.Response(200, json=_strong_tool_response("edit_file"))
        if "opencode" in url:
            call_count["opencode"] += 1
            return httpx.Response(200, json=_completion("Final answer from opencode"))
        return httpx.Response(404)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(url__regex=r"http://.*:.*/v1/chat/completions").mock(side_effect=_route)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "messages": _messages_with_last_assistant_tool("read_file"),
                },
            )

    # Cheap + minimax (escalation 1) + opencode (escalation 2 -- final, no strong tool).
    assert call_count["athena"] == 1
    # At most 2 escalations to strong backends total.
    total_strong = call_count["minimax"] + call_count["opencode"]
    assert total_strong <= 2
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Defensive: no strong backend available → 502 (not the cheap answer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_strong_backend_returns_502_not_cheap_answer():
    """If cheap returns a strong tool but no strong backend is reachable, return 502.

    The cheap answer MUST NOT leak to the client even on failure.
    """
    config = _adaptive_config()
    config["routing"]["adaptive_strong_backends"] = []
    config["routing"]["default_backends"] = ["athena"]

    app = create_app(config)
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        athena = mock.post("http://athena:8083/v1/chat/completions").respond(
            200, json=_strong_tool_response("write_file")
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "messages": _messages_with_last_assistant_tool("read_file"),
                },
            )

    assert athena.called
    assert response.status_code == 502
    body = response.text
    assert "write_file" not in body


# ---------------------------------------------------------------------------
# Non-adaptive config: no escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_adaptive_config_does_not_escalate():
    """With adaptive=False, the cheap response is forwarded as-is."""
    config = _adaptive_config()
    config["routing"]["adaptive"] = False

    app = create_app(config)
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        athena = mock.post("http://athena:8083/v1/chat/completions").respond(
            200, json=_strong_tool_response("write_file")
        )
        mock.post("http://minimax:8090/v1/chat/completions").respond(200, json=_completion("STRONG"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "messages": _text_only_messages(),
                },
            )

    assert response.status_code == 200
    assert athena.called
    assert response.json()["_backend"] == "athena"


# ---------------------------------------------------------------------------
# Vision requests skip adaptive routing entirely
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vision_request_skips_adaptive_routing():
    """Vision requests must not invoke adaptive escalate-and-retry."""
    config = _adaptive_config()
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)

    with respx.mock(assert_all_called=False) as mock:
        athena = mock.post("http://athena:8083/v1/chat/completions").respond(
            200, json=_completion("vision response")
        )
        mock.post("http://minimax:8090/v1/chat/completions").respond(200, json=_completion("strong"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3.6-35b-a3b-fast-256k",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "what is this?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "http://example.com/x.png"},
                                },
                            ],
                        }
                    ],
                },
            )

    assert response.status_code == 200
    assert athena.called
