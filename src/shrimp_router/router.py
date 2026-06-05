"""Request routing and fallback logic."""

from __future__ import annotations

import logging

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

from .backends import BackendManager
from .models import ChatCompletionRequest

logger = logging.getLogger(__name__)

# Escalation: if an agent is struggling (high loop count, no commits yet),
# override the normal routing and promote to the strongest available backend.
# Toggle via config: escalation.enabled (default true)
_ESCALATION_LOOP_THRESHOLD = 50   # loops before escalation kicks in
_ESCALATION_BACKENDS = ["minimax", "opencode", "kimi", "openrouter"]  # strongest first


def _escalation_candidates(manager: BackendManager, config: dict) -> list[str] | None:
    """Return override backend list if escalation is configured and enabled."""
    esc = config.get("escalation", {})
    if not esc.get("enabled", True):
        return None
    return esc.get("backends", _ESCALATION_BACKENDS)


def _should_escalate(request: Request, config: dict) -> bool:
    """True if this request should be escalated to a stronger model."""
    esc = config.get("escalation", {})
    if not esc.get("enabled", True):
        return False
    try:
        loop = int(request.headers.get("X-Loop-Count", 0))
        has_commits = request.headers.get("X-Has-Commits", "true").lower() == "true"
        threshold = int(esc.get("loop_threshold", _ESCALATION_LOOP_THRESHOLD))
        return loop >= threshold and not has_commits
    except Exception:
        return False


def _is_circuit_breaker_failure(status_code: int) -> bool:
    """Return True if the status code should be recorded as a circuit-breaker failure."""
    if status_code >= 500:
        return True
    if status_code == 429:
        return True
    return False


async def handle_chat(request: Request, body: ChatCompletionRequest) -> StreamingResponse | dict:
    """Route a chat completion request, with fallback on 429/error."""
    manager: BackendManager = request.app.state.backend_manager
    config: dict = request.app.state.config
    task_type = request.headers.get("X-Task-Type")
    phase = request.headers.get("X-Phase")
    is_vision = body.is_vision_request()
    stream = body.stream or False

    if _should_escalate(request, config):
        loop = request.headers.get("X-Loop-Count", "?")
        logger.info(f"[router] escalating (loop={loop}, no commits) -> strongest backend")
        candidates = _escalation_candidates(manager, config) or await manager.pick_backends(task_type, is_vision, phase=phase)
    else:
        candidates = await manager.pick_backends(task_type, is_vision, phase=phase)

    if not candidates:
        return {"error": "No backends configured", "status_code": 503}

    payload = _build_payload(body)
    last_error: str = "No backends available"

    for name in candidates:
        if name not in manager.backends:
            continue
        if not manager._quota.has_capacity(name):
            logger.info(f"[router] {name} quota exhausted, skipping")
            continue

        # Circuit breaker: skip backends whose breaker is blocking requests
        cb = manager.circuit_breaker(name)
        if cb is not None and not cb.should_allow_request():
            status = cb.status()
            retry_after = status.get("remaining_cooldown_seconds", 0)
            logger.info(f"[router] {name} circuit breaker open, skipping (retry in {retry_after:.0f}s)")
            last_error = f"{name}: circuit breaker open"
            continue

        try:
            resp = await manager.forward(name, payload, stream=stream)
        except httpx.TimeoutException:
            logger.warning(f"[router] {name} timed out")
            if cb is not None:
                cb.record_failure(is_timeout=True)
            last_error = f"{name}: timeout"
            continue
        except Exception as e:
            logger.warning(f"[router] {name} error: {e}")
            if cb is not None:
                cb.record_failure()
            last_error = f"{name}: {e}"
            continue

        if resp.status_code == 429:
            logger.warning(f"[router] {name} rate-limited, trying next")
            if cb is not None:
                cb.record_failure()
            last_error = f"{name}: 429"
            continue

        if not (200 <= resp.status_code < 300):
            logger.warning(f"[router] {name} returned {resp.status_code}")
            if cb is not None and _is_circuit_breaker_failure(resp.status_code):
                cb.record_failure()
            last_error = f"{name}: HTTP {resp.status_code}"
            if 400 <= resp.status_code < 500:
                break
            continue

        # Successful response
        if cb is not None:
            cb.record_success()
        logger.info(f"[router] served by {name} (task_type={task_type}, vision={is_vision})")

        if stream:
            return StreamingResponse(
                _stream_response(resp, name),
                media_type="text/event-stream",
                headers={"X-Backend": name},
            )

        data = resp.json()
        data["model"] = body.model
        data["_backend"] = name
        return data

    return {"error": f"All backends failed: {last_error}", "status_code": 502}


def _build_payload(body: ChatCompletionRequest) -> dict:
    """Convert a ChatCompletionRequest to a plain dict for forwarding."""
    result: dict = {
        "model": body.model,
        "messages": [msg.model_dump() for msg in body.messages],
    }
    if body.max_tokens is not None:
        result["max_tokens"] = body.max_tokens
    if body.temperature is not None:
        result["temperature"] = body.temperature
    if body.stream is not None:
        result["stream"] = body.stream
    if body.stop is not None:
        result["stop"] = body.stop
    if body.n is not None:
        result["n"] = body.n
    return result


async def _stream_response(resp: httpx.Response, backend_name: str):
    """Yield SSE chunks from a streaming HTTP response."""
    async for chunk in resp.aiter_bytes():
        if chunk:
            yield chunk
    await resp.aclose()
