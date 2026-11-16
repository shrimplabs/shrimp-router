"""max_inflight: a backend with that many /v1/messages requests open is skipped
for the next candidate (2026-11-12: athena's model server, buried under
requests max_concurrency never limited, stalled every diagnose call)."""

from __future__ import annotations

import httpx
import pytest
import respx

from shrimp_router.app import create_app


def _config():
    return {
        "backends": {
            "local": {"base_url": "http://local:8080/v1", "models": ["qwen"], "max_inflight": 1,
                      "format": "openai", "response_format": "openai"},
            "cloud": {"base_url": "http://cloud:8080/v1", "models": ["deepseek"],
                      "format": "openai", "response_format": "openai"},
        },
        "routing": {"default_backends": ["local", "cloud"]},
    }


def _completion(text: str) -> dict:
    return {"id": "c", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


_SSE = (b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n")


async def _post(app, stream=False):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/messages", json={
            "model": "x", "max_tokens": 8, "stream": stream,
            "messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.asyncio
async def test_a_backend_at_its_cap_overflows_to_the_next():
    app = create_app(_config())
    mgr = app.state.backend_manager
    mgr._inflight["local"] = 1  # one request already open
    with respx.mock(assert_all_called=False) as mock:
        local = mock.post("http://local:8080/v1/chat/completions").respond(200, json=_completion("local"))
        cloud = mock.post("http://cloud:8080/v1/chat/completions").respond(200, json=_completion("cloud"))
        resp = await _post(app)
    assert resp.status_code == 200 and resp.headers["X-Backend"] == "cloud"
    assert not local.called and cloud.called


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_the_slot_is_freed_when_the_response_ends(stream):
    app = create_app(_config())
    mgr = app.state.backend_manager
    with respx.mock(assert_all_called=False) as mock:
        if stream:
            mock.post("http://local:8080/v1/chat/completions").respond(
                200, content=_SSE, headers={"Content-Type": "text/event-stream"})
        else:
            mock.post("http://local:8080/v1/chat/completions").respond(200, json=_completion("local"))
        for _ in range(2):  # the second request finds the slot free again
            resp = await _post(app, stream=stream)
            assert resp.status_code == 200 and resp.headers["X-Backend"] == "local"
    assert mgr._inflight["local"] == 0


@pytest.mark.asyncio
async def test_a_failed_backend_frees_its_slot():
    app = create_app(_config())
    mgr = app.state.backend_manager
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://local:8080/v1/chat/completions").respond(500, json={"error": "boom"})
        mock.post("http://cloud:8080/v1/chat/completions").respond(200, json=_completion("cloud"))
        resp = await _post(app)
    assert resp.headers["X-Backend"] == "cloud"
    assert mgr._inflight["local"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500])
async def test_a_refused_stream_releases_its_connection(status):
    # 2026-11-13: OpenCode at its weekly cap answered every streamed request
    # with 429; each skipped response stayed open, holding a pooled connection,
    # until all 100 were gone and every backend "timed out" for 90 minutes.
    app = create_app(_config())
    mgr = app.state.backend_manager
    sent = []
    send = mgr._client.send

    async def recording_send(req, **kw):
        resp = await send(req, **kw)
        sent.append(resp)
        return resp

    mgr._client.send = recording_send
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://local:8080/v1/chat/completions").respond(status, json={"error": "quota"})
        mock.post("http://cloud:8080/v1/chat/completions").respond(
            200, content=_SSE, headers={"Content-Type": "text/event-stream"})
        resp = await _post(app, stream=True)
    assert resp.headers["X-Backend"] == "cloud"
    assert sent[0].is_closed
