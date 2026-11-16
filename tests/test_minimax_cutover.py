"""MiniMax cutover: at cutover_percent used, every backend on the MiniMax key
is skipped until the window resets, so the fallbacks take over before
MiniMax refuses (2026-11-12: the poller matched no row and never fired)."""

from __future__ import annotations

import time

from shrimp_router.quota import MinimaxQuotaPoller, QuotaRegistry


def _data(remaining_pct: float, resets_in: float = 3600) -> dict:
    return {"model_remains": [
        {"model_name": "video", "current_interval_remaining_percent": 0},
        {"model_name": "general", "current_interval_remaining_percent": remaining_pct,
         "end_time": int((time.time() + resets_in) * 1000)},
    ]}


def _poller():
    reg = QuotaRegistry()
    for name in ("minimax", "minimax-fast"):
        reg.register(name, 18000, 999999)
    return reg, MinimaxQuotaPoller("k", reg, cutover_percent=90, backend_names=["minimax", "minimax-fast"])


def test_below_cutover_keeps_minimax_in_rotation():
    reg, poller = _poller()
    poller._apply(_data(remaining_pct=11))  # 89% used
    assert reg.has_capacity("minimax") and reg.has_capacity("minimax-fast")


def test_at_cutover_every_backend_on_the_key_is_skipped_until_reset():
    reg, poller = _poller()
    poller._apply(_data(remaining_pct=10, resets_in=3600))  # 90% used
    assert not reg.has_capacity("minimax") and not reg.has_capacity("minimax-fast")
    assert reg.stats()["minimax"]["exhausted"] is True
    left = reg.get("minimax")._exhausted_until - time.monotonic()
    assert 3500 < left <= 3600


def test_window_reset_puts_minimax_back():
    reg, poller = _poller()
    poller._apply(_data(remaining_pct=5))
    poller._apply(_data(remaining_pct=100))
    assert reg.has_capacity("minimax") and reg.has_capacity("minimax-fast")


def test_unknown_response_changes_nothing():
    reg, poller = _poller()
    poller._apply(_data(remaining_pct=5))
    poller._apply({"model_remains": []})
    assert not reg.has_capacity("minimax")
