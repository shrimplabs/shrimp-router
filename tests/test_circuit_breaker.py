"""Tests for the circuit breaker mechanism."""

from __future__ import annotations

import time

import pytest

from shrimp_router.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def test_circuit_starts_closed():
    cb = CircuitBreaker("test-backend", failure_threshold=3, cooldown_seconds=60)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_trips_after_threshold_failures():
    cb = CircuitBreaker("test-backend", failure_threshold=3, cooldown_seconds=60)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    # failure_count is not reset until success is recorded or cooldown probe begins
    assert cb.failure_count == 3


def test_circuit_stays_closed_below_threshold():
    cb = CircuitBreaker("test-backend", failure_threshold=3, cooldown_seconds=60)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED


def test_circuit_transitions_to_half_open_after_cooldown():
    cb = CircuitBreaker("test-backend", failure_threshold=2, cooldown_seconds=0.05)
    # Trip the circuit
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Wait for cooldown
    time.sleep(0.06)

    # should_allow_request transitions to HALF_OPEN
    allowed = cb.should_allow_request()
    assert allowed is True
    assert cb.state == CircuitState.HALF_OPEN


def test_circuit_closes_after_successful_probe():
    cb = CircuitBreaker("test-backend", failure_threshold=2, cooldown_seconds=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    time.sleep(0.06)
    cb.should_allow_request()  # transition to HALF_OPEN
    cb.record_success()  # probe succeeds
    assert cb.state == CircuitState.CLOSED


def test_circuit_reopens_after_failed_probe():
    cb = CircuitBreaker("test-backend", failure_threshold=2, cooldown_seconds=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    time.sleep(0.06)
    cb.should_allow_request()  # transition to HALF_OPEN
    cb.record_failure()  # probe fails
    assert cb.state == CircuitState.OPEN


def test_circuit_resets_failure_count_on_success():
    cb = CircuitBreaker("test-backend", failure_threshold=3, cooldown_seconds=60)
    cb.record_failure()
    cb.record_failure()
    assert cb.failure_count == 2

    cb.record_success()  # resets counter
    assert cb.failure_count == 0

    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Fail-fast behaviour
# ---------------------------------------------------------------------------

def test_open_circuit_blocks_requests():
    cb = CircuitBreaker("test-backend", failure_threshold=2, cooldown_seconds=10)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    allowed = cb.should_allow_request()
    assert allowed is False


def test_open_circuit_returns_remaining_cooldown():
    cb = CircuitBreaker("test-backend", failure_threshold=1, cooldown_seconds=60)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    status = cb.status()
    assert status["state"] == "open"
    assert 59 < status["remaining_cooldown_seconds"] <= 60


def test_circuit_allows_request_when_closed():
    cb = CircuitBreaker("test-backend", failure_threshold=3, cooldown_seconds=60)
    assert cb.should_allow_request() is True


# ---------------------------------------------------------------------------
# Per-backend isolation
# ---------------------------------------------------------------------------

def test_different_backends_have_independent_circuits():
    cb1 = CircuitBreaker("backend-a", failure_threshold=2, cooldown_seconds=60)
    cb2 = CircuitBreaker("backend-b", failure_threshold=2, cooldown_seconds=60)

    # Trip backend-a
    cb1.record_failure()
    cb1.record_failure()
    assert cb1.state == CircuitState.OPEN

    # backend-b stays closed
    assert cb2.state == CircuitState.CLOSED

    # can_request on backend-b still works
    assert cb2.should_allow_request() is True

    # backend-a still blocked
    assert cb1.should_allow_request() is False


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_default_config():
    cb = CircuitBreaker("test")
    assert cb.failure_threshold == 3
    assert cb.cooldown_seconds == 60


def test_custom_config():
    cb = CircuitBreaker("test", failure_threshold=10, cooldown_seconds=120)
    assert cb.failure_threshold == 10
    assert cb.cooldown_seconds == 120


# ---------------------------------------------------------------------------
# CircuitBreakerRegistry
# ---------------------------------------------------------------------------

def test_registry_creates_breaker():
    reg = CircuitBreakerRegistry()
    cb = reg.register("minimax", {"failure_threshold": 5, "cooldown_seconds": 30})
    assert cb.name == "minimax"
    assert cb.failure_threshold == 5
    assert cb.cooldown_seconds == 30


def test_registry_returns_existing_breaker():
    reg = CircuitBreakerRegistry()
    cb1 = reg.register("minimax", {"failure_threshold": 5})
    cb2 = reg.register("minimax", {"failure_threshold": 99})  # should not overwrite
    assert cb1 is cb2
    assert cb1.failure_threshold == 5


def test_registry_get_returns_none_for_unknown():
    reg = CircuitBreakerRegistry()
    assert reg.get("unknown") is None


def test_registry_status_all():
    reg = CircuitBreakerRegistry()
    reg.register("backend-a", {"failure_threshold": 3})
    reg.register("backend-b", {"failure_threshold": 5})
    status = reg.status_all()
    assert "backend-a" in status
    assert "backend-b" in status
    assert status["backend-a"]["failure_threshold"] == 3
    assert status["backend-b"]["failure_threshold"] == 5


# ---------------------------------------------------------------------------
# Status dict
# ---------------------------------------------------------------------------

def test_status_includes_all_fields():
    cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
    status = cb.status()
    assert "state" in status
    assert "failure_count" in status
    assert "failure_threshold" in status
    assert "cooldown_seconds" in status
    assert "remaining_cooldown_seconds" in status
    assert status["state"] == "closed"


def test_status_remaining_cooldown_decreases_over_time():
    cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=2)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    initial = cb.status()["remaining_cooldown_seconds"]
    time.sleep(0.5)
    later = cb.status()["remaining_cooldown_seconds"]
    assert later < initial


def test_is_open_property():
    cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=60)
    assert cb.is_open() is False  # CLOSED
    cb.record_failure()
    assert cb.is_open() is True   # OPEN
    # HALF_OPEN is not considered "open" -- is_open() returns False
    cb._state = CircuitState.HALF_OPEN
    assert cb.is_open() is False
