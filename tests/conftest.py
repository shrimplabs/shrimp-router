"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest
import respx

from shrimp_router.models import RouterConfig


@pytest.fixture
def router_config() -> RouterConfig:
    """RouterConfig with two mock backends."""
    return RouterConfig.model_validate({
        "backends": {
            "powerful-mini": {
                "base_url": "http://mini-1:8080/v1",
                "models": ["llama3.1"],
                "max_concurrency": 1,
            },
            "fast-mini": {
                "base_url": "http://mini-2:8080/v1",
                "models": ["llama3.2"],
                "max_concurrency": 1,
            },
        },
        "routing": {
            "default_backends": ["powerful-mini", "fast-mini"],
        },
    })


@pytest.fixture
def app_with_mocked_backends():
    """Create app with real backend_manager but all HTTP mocked via respx."""
    from shrimp_router.app import create_app

    config = {
        "backends": {
            "powerful-mini": {
                "base_url": "http://mini-1:8080/v1",
                "models": ["llama3.1"],
                "max_concurrency": 1,
            },
            "fast-mini": {
                "base_url": "http://mini-2:8080/v1",
                "models": ["llama3.2"],
                "max_concurrency": 1,
            },
        },
        "routing": {
            "default_backends": ["powerful-mini", "fast-mini"],
        },
    }
    return create_app(config)


@pytest.fixture
def client(app_with_mocked_backends):
    """FastAPI TestClient for integration tests."""
    from fastapi.testclient import TestClient

    return TestClient(app_with_mocked_backends)


@pytest.fixture
def respx_mock_session():
    """Global respx mock that patches httpx.AsyncClient for all backends."""
    with respx.mock(assert_all_called=False) as mock:
        yield mock
