"""Bidirectional response-format normalizer.

Translates between OpenAI /chat/completions and Anthropic /messages response
schemas so callers always receive the format matching their request endpoint,
regardless of the backend wire format.

All functions are pure (no I/O).
"""
from __future__ import annotations


from typing import Any



# ---------------------------------------------------------------------------
# OpenAI → Anthropic
# ---------------------------------------------------------------------------


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
    content_text = message.get("content", "")
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
