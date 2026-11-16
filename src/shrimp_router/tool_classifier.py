"""Classify the last assistant turn as read-only, write/strong, or unknown.

Used by adaptive routing to decide whether a request can be offloaded to a
cheap local backend (read-only) or must go to a strong/billed backend
(write/execute). Inspects both Anthropic-style ``tool_use`` content blocks
and OpenAI-style ``tool_calls`` arrays.
"""

from __future__ import annotations

from typing import Any, Iterable


# Canonical read-only tool names. Comparison is case-insensitive and
# whitespace-tolerant (see ``_normalize``), so callers may pass any casing.
READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "read_file_range",
    "list_files",
    "search_code",
})

# Canonical strong/write tool names. If ANY tool in the last assistant turn
# resolves to one of these, the turn is classified as "write".
STRONG_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "patch_file",
    "bash",
    "run_command",
    "computer",
    "str_replace_editor",
    "create_file",
    "delete_file",
    "execute_command",
    "terminal",
})


def _normalize(name: Any) -> str:
    """Normalize a tool name for comparison: lowercase, stripped."""
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _extract_tool_names(message: dict[str, Any]) -> list[str]:
    """Return normalized tool names referenced by an assistant message.

    Supports two wire formats:

    * Anthropic ``/messages`` style -- ``content`` is a list of content
      blocks. Blocks with ``type == "tool_use"`` expose the tool name via a
      top-level ``name`` field.
    * OpenAI ``/chat/completions`` style -- the assistant ``message`` carries
      a ``tool_calls`` list, each entry shaped like
      ``{"function": {"name": ..., "arguments": ...}}``.
    """
    names: list[str] = []

    # --- OpenAI-style tool_calls on the assistant message itself ---
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function")
            if isinstance(func, dict):
                names.append(_normalize(func.get("name", "")))

    # --- Anthropic-style content blocks (and OpenAI tool_calls inside content) ---
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                names.append(_normalize(block.get("name", "")))
            elif btype == "tool_calls":
                # Some OpenAI-compatible backends nest tool_calls inside a
                # content block instead of on the message itself.
                nested = block.get("tool_calls")
                if isinstance(nested, list):
                    for tc in nested:
                        if not isinstance(tc, dict):
                            continue
                        func = tc.get("function")
                        if isinstance(func, dict):
                            names.append(_normalize(func.get("name", "")))
    elif isinstance(content, str):
        # Plain text content -- no tool invocations.
        pass

    return names


def classify_last_tool(messages: Iterable[dict[str, Any] | Any]) -> str:
    """Classify the last assistant turn in ``messages``.

    Returns one of:

    * ``"write"``   -- at least one tool in ``STRONG_TOOLS`` was invoked.
    * ``"read"``    -- at least one tool was invoked and all are in
      ``READ_ONLY_TOOLS``.
    * ``"unknown"`` -- no assistant turn, no tool invocations, or only
      unrecognised tool names.

    The function never raises on malformed input; defensive guards keep it
    safe to call with partially-validated message lists.
    """
    if messages is None:
        return "unknown"

    try:
        message_list = list(messages)
    except TypeError:
        return "unknown"

    if not message_list:
        return "unknown"

    # Walk messages in reverse to find the LAST assistant turn.
    last_assistant: dict[str, Any] | None = None
    for msg in reversed(message_list):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            last_assistant = msg
            break

    if last_assistant is None:
        return "unknown"

    names = _extract_tool_names(last_assistant)
    if not names:
        return "unknown"

    # Write dominates: any strong tool makes the turn "write".
    has_strong = any(n in STRONG_TOOLS for n in names)
    if has_strong:
        return "write"

    has_read = any(n in READ_ONLY_TOOLS for n in names)
    if has_read:
        return "read"

    return "unknown"
