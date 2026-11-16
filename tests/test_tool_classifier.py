"""Tests for the tool_classifier module."""

from __future__ import annotations

import pytest

from shrimp_router.tool_classifier import (
    READ_ONLY_TOOLS,
    STRONG_TOOLS,
    classify_last_tool,
)


# ---------------------------------------------------------------------------
# Anthropic-style tool_use content blocks
# ---------------------------------------------------------------------------


def test_anthropic_read_only_tool_use():
    """Anthropic tool_use block with a read-only name returns 'read'."""
    messages = [
        {"role": "user", "content": "show me foo.py"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "read_file",
                    "input": {"path": "foo.py"},
                }
            ],
        },
    ]
    assert classify_last_tool(messages) == "read"


def test_anthropic_strong_tool_use():
    """Anthropic tool_use block with a write tool returns 'write'."""
    messages = [
        {"role": "user", "content": "add a function"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "write_file",
                    "input": {"path": "foo.py", "content": "def f(): pass\n"},
                }
            ],
        },
    ]
    assert classify_last_tool(messages) == "write"


def test_anthropic_mixed_text_and_tool_use():
    """Content blocks with mixed text + tool_use: only tool_use blocks count."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the file."},
                {
                    "type": "tool_use",
                    "name": "read_file",
                    "input": {"path": "foo.py"},
                },
            ],
        }
    ]
    assert classify_last_tool(messages) == "read"


# ---------------------------------------------------------------------------
# OpenAI-style tool_calls
# ---------------------------------------------------------------------------


def test_openai_tool_calls_read():
    """OpenAI tool_calls with a read tool return 'read'."""
    messages = [
        {"role": "user", "content": "find usages"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "arguments": '{"pattern": "foo"}',
                    },
                }
            ],
        },
    ]
    assert classify_last_tool(messages) == "read"


def test_openai_tool_calls_write():
    """OpenAI tool_calls with a write tool return 'write'."""
    messages = [
        {"role": "user", "content": "run tests"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"cmd": "pytest"}',
                    },
                }
            ],
        },
    ]
    assert classify_last_tool(messages) == "write"


# ---------------------------------------------------------------------------
# Precedence & ambiguity
# ---------------------------------------------------------------------------


def test_mixed_read_and_write_write_wins():
    """When both read and write tools are present, 'write' wins."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "read_file",
                    "input": {"path": "foo.py"},
                },
                {
                    "type": "tool_use",
                    "name": "edit_file",
                    "input": {"path": "foo.py", "old": "a", "new": "b"},
                },
            ],
        }
    ]
    assert classify_last_tool(messages) == "write"


def test_unknown_tool_name():
    """An unrecognised tool name returns 'unknown'."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "weird_tool",
                    "input": {},
                }
            ],
        }
    ]
    assert classify_last_tool(messages) == "unknown"


def test_case_insensitive_strong():
    """Tool name comparison is case-insensitive."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "BASH", "input": {}}
            ],
        }
    ]
    assert classify_last_tool(messages) == "write"


def test_case_insensitive_read():
    """Uppercase read tool still classifies as 'read'."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "LIST_FILES", "input": {}}
            ],
        }
    ]
    assert classify_last_tool(messages) == "read"


def test_whitespace_around_tool_name():
    """Leading/trailing whitespace is tolerated."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "  read_file  ", "input": {}}
            ],
        }
    ]
    assert classify_last_tool(messages) == "read"


# ---------------------------------------------------------------------------
# Empty / defensive cases
# ---------------------------------------------------------------------------


def test_no_assistant_message():
    """Messages with no assistant turn return 'unknown'."""
    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "hello"},
    ]
    assert classify_last_tool(messages) == "unknown"


def test_empty_messages_list():
    """Empty message list returns 'unknown' and does not raise."""
    assert classify_last_tool([]) == "unknown"


def test_none_messages():
    """None input returns 'unknown' and does not raise."""
    assert classify_last_tool(None) == "unknown"


def test_assistant_without_tool_calls():
    """Assistant turn with text-only content returns 'unknown'."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here is the answer."}
            ],
        }
    ]
    assert classify_last_tool(messages) == "unknown"


def test_non_dict_messages_guarded():
    """Non-dict entries in the messages list are skipped without raising."""
    messages = [
        "not a dict",
        42,
        None,
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "read_file", "input": {}}
            ],
        },
    ]
    assert classify_last_tool(messages) == "read"


def test_non_dict_messages_only_returns_unknown():
    """A messages list with no dicts at all returns 'unknown'."""
    assert classify_last_tool(["a", "b", None]) == "unknown"


# ---------------------------------------------------------------------------
# Last-assistant semantics
# ---------------------------------------------------------------------------


def test_walks_messages_in_reverse_to_find_last_assistant():
    """Only the LAST assistant turn is considered."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "write_file", "input": {}}
            ],
        },
        {"role": "user", "content": "thanks"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "read_file", "input": {}}
            ],
        },
    ]
    # Last assistant is read-only -> 'read'.
    assert classify_last_tool(messages) == "read"


# ---------------------------------------------------------------------------
# Sanity checks on the public sets
# ---------------------------------------------------------------------------


def test_read_only_and_strong_tools_are_disjoint():
    """A tool cannot be both read-only and strong."""
    overlap = READ_ONLY_TOOLS & STRONG_TOOLS
    assert overlap == set()


@pytest.mark.parametrize(
    "name",
    sorted(READ_ONLY_TOOLS),
)
def test_every_read_only_tool_classifies_as_read(name):
    """Each name in READ_ONLY_TOOLS classifies as 'read' on its own."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": name, "input": {}}
            ],
        }
    ]
    assert classify_last_tool(messages) == "read"


@pytest.mark.parametrize(
    "name",
    sorted(STRONG_TOOLS),
)
def test_every_strong_tool_classifies_as_write(name):
    """Each name in STRONG_TOOLS classifies as 'write' on its own."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": name, "input": {}}
            ],
        }
    ]
    assert classify_last_tool(messages) == "write"
