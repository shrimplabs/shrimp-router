"""Bidirectional response-format normalizer.

Translates between OpenAI /chat/completions and Anthropic /messages response
schemas so callers always receive the format matching their request endpoint,
regardless of the backend wire format.

All functions are pure (no I/O) except openai_stream_to_anthropic, which
reads the backend stream it is given.
"""
from __future__ import annotations

import json
from typing import Any



# ---------------------------------------------------------------------------
# OpenAI → Anthropic
# ---------------------------------------------------------------------------


def _split_content(content) -> tuple[str, str]:
    """(thinking, text) from an OpenAI content field. Mistral sends typed part
    lists for reasoning models -- [{"type": "thinking", "thinking": [{"type":
    "text", "text": ...}]}, {"type": "text", "text": ...}] -- where others send
    a string; reading only strings made every GLM reply empty."""
    if isinstance(content, str):
        return "", content
    thinking, text = [], []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text.append(part.get("text") or "")
        elif part.get("type") == "thinking":
            inner = part.get("thinking")
            if isinstance(inner, str):
                thinking.append(inner)
            elif isinstance(inner, list):
                thinking.extend(p.get("text") or "" for p in inner if isinstance(p, dict))
    return "".join(thinking), "".join(text)


def openai_to_anthropic_response(openai_resp: dict[str, Any], original_model: str) -> dict[str, Any]:
    """Translate an OpenAI /chat/completions response to Anthropic /messages response shape.

    Required fields preserved:
      - id          → id
      - model       → model
      - type        → "message"
      - choices[0].message.content  → content[0].text
      - choices[0].finish_reason     → stop_reason
      - usage.prompt_tokens          → usage.input_tokens
      - usage.completion_tokens      → usage.output_tokens
    """
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    _thinking, content_text = _split_content(message.get("content") or "")
    finish_reason = choice.get("finish_reason", "stop")


    # Map OpenAI finish_reason → Anthropic stop_reason
    stop_reason_map: dict[str, str] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")


    usage = openai_resp.get("usage", {})

    return {
        "id": openai_resp.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": [{"type": "text", "text": content_text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Anthropic → OpenAI
# ---------------------------------------------------------------------------



def anthropic_to_openai_response(anthropic_resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an Anthropic /messages response to OpenAI /chat/completions response shape.

    Required fields preserved:
      - id              → id
      - model           → model
      - type="message"  → dropped (OpenAI has no equivalent)
      - content[0].text → choices[0].message.content
      - stop_reason     → choices[0].finish_reason
      - usage.input_tokens     → usage.prompt_tokens
      - usage.output_tokens    → usage.completion_tokens
    """
    # Extract text from Anthropic content blocks
    content_text = ""
    for block in anthropic_resp.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            content_text = block.get("text", "")
            break

    stop_reason = anthropic_resp.get("stop_reason", "stop")
    # Map Anthropic stop_reason → OpenAI finish_reason
    finish_reason_map: dict[str, str] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
    }
    finish_reason = finish_reason_map.get(stop_reason, "stop")

    usage = anthropic_resp.get("usage", {})


    return {
        "id": anthropic_resp.get("id", ""),
        "object": "chat.completion",
        "created": 0,  # Anthropic responses don't include "created"; caller may override
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content_text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            ),
        },
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def openai_stream_to_anthropic(resp, original_model: str):
    """Re-emit an OpenAI /chat/completions SSE stream as Anthropic /messages SSE.

    The /v1/messages endpoint used to pass an OpenAI-format backend's stream
    through untranslated, so Anthropic clients (the swarm streams every call)
    saw no content_block events: 557/557 calls to athena's qwen-flash-next came
    back empty. Reasoning deltas (reasoning_content / reasoning) become a
    thinking block, content deltas a text block; usage comes from the final
    chunk (requested with stream_options.include_usage).
    """
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": "", "type": "message", "role": "assistant", "model": original_model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}})
    index = -1
    open_kind = None
    stop_reason = "end_turn"
    usage: dict[str, Any] = {}
    stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                "content_filter": "stop_sequence"}

    def switch(kind: str):
        nonlocal index, open_kind
        out = b""
        if open_kind is not None:
            out += _sse("content_block_stop", {"type": "content_block_stop", "index": index})
        index += 1
        open_kind = kind
        block = {"type": "thinking", "thinking": ""} if kind == "thinking" else {"type": "text", "text": ""}
        return out + _sse("content_block_start", {"type": "content_block_start", "index": index,
                                                  "content_block": block})

    buffer = ""
    try:
        async for chunk in resp.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    part_thinking, text = _split_content(delta.get("content") or "")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or part_thinking
                    if reasoning:
                        if open_kind != "thinking":
                            yield switch("thinking")
                        yield _sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                           "delta": {"type": "thinking_delta", "thinking": reasoning}})
                    if text:
                        if open_kind != "text":
                            yield switch("text")
                        yield _sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                                           "delta": {"type": "text_delta", "text": text}})
                    if choice.get("finish_reason"):
                        stop_reason = stop_map.get(choice["finish_reason"], "end_turn")
    finally:
        await resp.aclose()
    if open_kind is not None:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})
    yield _sse("message_delta", {"type": "message_delta",
                                 "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                 "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                                           "output_tokens": usage.get("completion_tokens", 0)}})
    yield _sse("message_stop", {"type": "message_stop"})
