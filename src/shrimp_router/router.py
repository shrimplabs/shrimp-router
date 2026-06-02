"""Request routing and fallback logic."""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

from .backends import BackendManager
from .models import ChatCompletionRequest

logger = logging.getLogger(__name__)


async def handle_chat(request: Request, body: ChatCompletionRequest) -> StreamingResponse | dict:
    """Route a chat completion request, with fallback on 429/error."""
    manager: BackendManager = request.app.state.backend_manager
    task_type = request.headers.get("X-Task-Type")
    is_vision = body.is_vision_request()
    stream = body.stream or False

    candidates = manager.pick_backends(task_type, is_vision)
    if not candidates:
        return {"error": "No backends configured", "status_code": 503}

    payload = _build_payload(body)
    last_error: str = "No backends available"

    for name in candidates:
        if not manager._quota.has_capacity(name):
            logger.info(f"[router] {name} quota exhausted, skipping")
            continue

        try:
            resp = await manager.forward(name, payload, stream=stream)
        except httpx.TimeoutException:
            logger.warning(f"[router] {name} timed out")
            last_error = f"{name}: timeout"
            continue
        except Exception as e:
            logger.warning(f"[router] {name} error: {e}")
            last_error = f"{name}: {e}"
            continue

        if resp.status_code == 429:
            logger.warning(f"[router] {name} rate-limited, trying next")
            last_error = f"{name}: 429"
            continue

        if not (200 <= resp.status_code < 300):
            logger.warning(f"[router] {name} returned {resp.status_code}")
            last_error = f"{name}: HTTP {resp.status_code}"
            # Don't retry on 4xx client errors
            if 400 <= resp.status_code < 500:
                break
            continue

        logger.info(f"[router] served by {name} (task_type={task_type}, vision={is_vision})")

        if stream:
            return StreamingResponse(
                _stream_response(resp, name),
                media_type="text/event-stream",
                headers={"X-Backend": name},
            )

        data = resp.json()
        data["model"] = body.model  # echo back requested model name
        data["_backend"] = name
        return data

    return {"error": f"All backends failed: {last_error}", "status_code": 502}


def _build_payload(body: ChatCompletionRequest) -> dict:
    """Serialize request, dropping None fields."""
    d = body.model_dump(mode="json", exclude_none=True)
    d.pop("extra_body", None)
    # Merge extra_body fields in if present
    if body.extra_body:
        d.update(body.extra_body)
    return d


async def _stream_response(resp: httpx.Response, backend_name: str):
    """Proxy SSE stream from backend to client."""
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()
