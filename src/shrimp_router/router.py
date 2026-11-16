"""Request routing and fallback logic."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

from .backends import BackendManager
from .format_normalizer import anthropic_to_openai_response
from .models import ChatCompletionRequest
from .tool_classifier import STRONG_TOOLS, classify_last_tool

logger = logging.getLogger(__name__)

# Escalation: if an agent is struggling (high loop count, no commits yet),
# override the normal routing and promote to the strongest available backend.
# Toggle via config: escalation.enabled (default true)
_ESCALATION_LOOP_THRESHOLD = 50   # loops before escalation kicks in
_ESCALATION_BACKENDS = ["minimax", "opencode", "kimi", "openrouter"]  # strongest first

# Adaptive routing: cap how many escalate-and-retry hops we'll do per request
# before giving up. The cheap response must NEVER reach the client on
# escalation, so we always re-call rather than forwarding the cheap answer.
_MAX_ADAPTIVE_ESCALATIONS = 2


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


def _response_has_strong_tool(data: dict) -> bool:
    """True if the OpenAI/Anthropic-shaped response includes a strong tool call.

    Inspects:
    * OpenAI ``choices[*].message.tool_calls[*].function.name``
    * Anthropic ``content[*].tool_use.name``
    * Some backends nest OpenAI tool_calls inside a content block; handle that too.
    """
    if not isinstance(data, dict):
        return False

    def _check_name(name: Any) -> bool:
        if not isinstance(name, str):
            return False
        return name.strip().lower() in STRONG_TOOLS

    choices = data.get("choices") or []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        if not isinstance(msg, dict):
            continue
        # OpenAI-style tool_calls on the message
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function")
                if isinstance(func, dict) and _check_name(func.get("name", "")):
                    return True
        # Anthropic-style content blocks
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use" and _check_name(block.get("name", "")):
                    return True
                if btype == "tool_calls":
                    nested = block.get("tool_calls")
                    if isinstance(nested, list):
                        for tc in nested:
                            if not isinstance(tc, dict):
                                continue
                            func = tc.get("function")
                            if isinstance(func, dict) and _check_name(func.get("name", "")):
                                return True

    # Anthropic top-level content field (when response isn't wrapped in choices)
    top_content = data.get("content")
    if isinstance(top_content, list):
        for block in top_content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if _check_name(block.get("name", "")):
                    return True

    return False


async def _try_backend(
    manager: BackendManager,
    name: str,
    payload: dict,
    stream: bool,
    model: str,
    incoming_auth: str | None = None,
) -> tuple[httpx.Response | None, str | None]:
    """Try a single backend. Returns (response, error_reason).

    On success, response is the raw httpx.Response (200). On failure, response
    is None and error_reason is a short string suitable for the last_error
    accumulator. Records circuit-breaker failures/successes as a side effect.
    """
    if name not in manager.backends:
        return None, f"{name}: not configured"
    if not manager._quota.has_capacity(name):
        logger.info(f"[router] {name} quota exhausted, skipping")
        return None, f"{name}: quota exhausted"

    cb = manager.circuit_breaker(name)
    if cb is not None and not cb.should_allow_request():
        status = cb.status()
        retry_after = status.get("remaining_cooldown_seconds", 0)
        logger.info(f"[router] {name} circuit breaker open, skipping (retry in {retry_after:.0f}s)")
        return None, f"{name}: circuit breaker open"

    try:
        resp = await manager.forward(name, payload, stream=stream, incoming_auth=incoming_auth)
    except httpx.TimeoutException:
        logger.warning(f"[router] {name} timed out")
        if cb is not None:
            cb.record_failure()
        return None, f"{name}: timeout"
    except Exception as e:
        logger.warning(f"[router] {name} error: {e}")
        if cb is not None:
            cb.record_failure()
        return None, f"{name}: {e}"

    if resp.status_code == 429:
        logger.warning(f"[router] {name} rate-limited, trying next")
        if cb is not None:
            cb.record_failure()
        return None, f"{name}: 429"

    if not (200 <= resp.status_code < 300):
        logger.warning(f"[router] {name} returned {resp.status_code}")
        if cb is not None and _is_circuit_breaker_failure(resp.status_code):
            cb.record_failure()
        return None, f"{name}: HTTP {resp.status_code}"

    # Success
    if cb is not None:
        cb.record_success()
    return resp, None


async def handle_chat(request: Request, body: ChatCompletionRequest) -> StreamingResponse | dict:
    """Route a chat completion request, with fallback on 429/error.

    Adaptive routing: when ``routing.adaptive`` is enabled and the request is
    a non-vision text request, we ask ``BackendManager.pick_backends`` to
    seed the candidate list with the cheap backend (``tool_class == 'read'``)
    or strong backend (``tool_class == 'write' | 'unknown'``). After a cheap
    response comes back, if the response contains a strong-tool call, we
    DISCARD it and re-call a strong backend. The cheap response never reaches
    the client. Streaming responses skip escalation entirely.
    """
    manager: BackendManager = request.app.state.backend_manager
    config: dict = request.app.state.config
    task_type = request.headers.get("X-Task-Type")
    phase = request.headers.get("X-Phase")
    is_vision = body.is_vision_request()
    stream = body.stream or False

    # Adaptive routing is opt-in via routing.adaptive. For vision or streamed
    # requests we keep the legacy path (no escalate-and-retry).
    adaptive_enabled = bool(getattr(manager.routing, "adaptive", False))

    # Read raw message dicts from the request body. We deliberately bypass
    # ``body.messages`` here because the strict ``ChatMessage`` Pydantic
    # model strips tool-use metadata (Anthropic ``content[*].tool_use.name``
    # / OpenAI ``tool_calls``) which the classifier needs to see.
    raw_messages: list[dict] = []
    try:
        raw_body = await request.body()
        if raw_body:
            import json as _json
            parsed = _json.loads(raw_body)
            if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
                raw_messages = [m for m in parsed["messages"] if isinstance(m, dict)]
    except Exception:
        raw_messages = [m.model_dump() for m in body.messages]

    if _should_escalate(request, config):
        loop = request.headers.get("X-Loop-Count", "?")
        logger.info(f"[router] escalating (loop={loop}, no commits) -> strongest backend")
        candidates = _escalation_candidates(manager, config) or await manager.pick_backends(task_type, is_vision, phase=phase)
    else:
        # Classify the last assistant turn to drive adaptive cheap/strong pick.
        tool_class: str | None = None
        if adaptive_enabled and not is_vision:
            try:
                tool_class = classify_last_tool(raw_messages)
            except Exception:
                tool_class = None
        candidates = await manager.pick_backends(task_type, is_vision, phase=phase, tool_class=tool_class)

    if not candidates:
        return {"error": "No backends configured", "status_code": 503}

    payload = _build_payload(body)
    last_error: str = "No backends available"

    # Cheap-list candidate index range -- used to detect when we're falling
    # back out of the cheap list. Strong backends come AFTER the cheap list
    # in ``adaptive_strong_backends``; for non-adaptive configs this is just
    # the default backend order.
    routing_cfg = manager.routing
    adaptive_strong = list(getattr(routing_cfg, "adaptive_strong_backends", []) or [])
    cheap_set = set(getattr(routing_cfg, "adaptive_cheap_backends", []) or [])

    # Used to prevent repeated re-entry into the cheap list on escalation
    # rounds. We track which backends we've already exhausted.
    tried: set[str] = set()
    escalations_used = 0

    # The current candidate list to walk; this is rebuilt when we need to
    # fall back to strong backends after a cheap response with a write tool.
    current_candidates = list(candidates)

    while current_candidates:
        progressed = False
        for name in current_candidates:
            if name in tried:
                continue
            tried.add(name)

            resp, err = await _try_backend(manager, name, payload, stream=stream, model=body.model, incoming_auth=request.headers.get("Authorization"))
            if err is not None:
                last_error = err
                # 4xx is a hard client error -- stop trying further backends
                if err.endswith(")") and "HTTP 4" in err:
                    return {"error": f"All backends failed: {last_error}", "status_code": 502}
                continue

            assert resp is not None
            logger.info(
                f"[router] served by {name} (task_type={task_type}, vision={is_vision})"
            )

            # Streaming: skip escalate-and-retry; return what we have.
            if stream:
                return StreamingResponse(
                    _stream_response(resp, name),
                    media_type="text/event-stream",
                    headers={"X-Backend": name},
                )

            data = resp.json()

            # Normalize Anthropic-shaped response to OpenAI format if needed.
            cfg = manager.backends.get(name)
            if cfg and cfg.effective_response_format == "anthropic":
                data = anthropic_to_openai_response(data, body.model)

            # Adaptive escalate-and-retry: if we just served from the cheap
            # list and the response asks for a strong tool, drop this
            # answer and try a strong backend instead. Never forward the
            # cheap response.
            if (
                adaptive_enabled
                and not is_vision
                and name in cheap_set
                and escalations_used < _MAX_ADAPTIVE_ESCALATIONS
                and _response_has_strong_tool(data)
            ):
                escalations_used += 1
                logger.info(
                    f"[router] escalating (cheap={name} returned strong tool) -> strong backend"
                )
                # Build the strong candidate list -- prefer configured strong
                # backends; if those are exhausted, fall through to anything
                # not yet tried.
                strong_candidates: list[str] = []
                for s in adaptive_strong:
                    if s in manager.backends and s not in tried:
                        strong_candidates.append(s)
                if not strong_candidates:
                    for s in manager.backends.keys():
                        if s not in tried and s not in cheap_set:
                            strong_candidates.append(s)
                if not strong_candidates:
                    # Nothing strong available -- do not poison client with
                    # cheap answer; return 502 with explanation.
                    return {
                        "error": f"Adaptive escalation needed but no strong backend available (last_error={last_error})",
                        "status_code": 502,
                    }
                current_candidates = strong_candidates
                progressed = True
                break  # restart the while loop with the strong list

            data["model"] = body.model
            data["_backend"] = name
            return data

        # If we made progress (escalation), loop again with the new list.
        if progressed:
            continue
        # Exhausted all current candidates without escalation.
        break

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
