"""Per-session backends (OpenCode Go) get a stable session id and a client
user agent; others get nothing extra (2026-11-12: Go returned HTTP 400
MissingSessionID for every request without x-opencode-session)."""
from types import SimpleNamespace

from shrimp_router.app import _client_headers
from shrimp_router.models import BackendConfig


def _req(headers):
    return SimpleNamespace(headers=headers)


def _cfg(**kw):
    return BackendConfig(base_url="http://localhost:8886/v1", models=["kimi-k2.6"], **kw)


def test_session_header_uses_the_callers_session_id():
    cfg = _cfg(session_header="x-opencode-session", user_agent="shrimphub-swarm/1.0")
    got = _client_headers(cfg, _req({"X-Session-Id": "swarm-task-7"}), {"messages": []})
    assert got == {"x-opencode-session": "swarm-task-7", "User-Agent": "shrimphub-swarm/1.0"}


def test_session_id_falls_back_to_a_stable_conversation_hash():
    cfg = _cfg(session_header="x-opencode-session")
    first = {"system": "sys", "messages": [{"role": "user", "content": "task"}]}
    later = {"system": "sys", "messages": first["messages"] + [{"role": "assistant", "content": "ok"}]}
    other = {"system": "sys", "messages": [{"role": "user", "content": "another task"}]}
    a, b, c = (_client_headers(cfg, _req({}), p)["x-opencode-session"] for p in (first, later, other))
    assert a == b != c


def test_plain_backends_get_no_extra_headers():
    assert _client_headers(_cfg(), _req({"X-Session-Id": "s"}), {"messages": []}) == {}
