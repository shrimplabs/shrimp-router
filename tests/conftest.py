"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest
import respx

from shrimp_vision_router.backends import BackendManager
from shrimp_vision_router.models import BackendConfig


@pytest.fixture
def backend_manager() -> BackendManager:
    """BackendManager with two mock backends, no real HTTP."""
    backends = {
        "powerful-mini": BackendConfig(
            base_url="http://mini-1:8080/v1",
            models=["llama3.1"],
            max_concurrency=1,
        ),
        "fast-mini": BackendConfig(
            base_url="http://mini-2:8080/v1",
            models=["llama3.2"],
            max_concurrency=1,
        ),
    }
    return BackendManager(backends, request_timeout_seconds=10)


@pytest.fixture
def app_with_mocked_backends(backend_manager):
    """Create app with real backend_manager but all HTTP mocked via respx."""
    from shrimp_vision_router.app import create_app

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
        }
    }
    app = create_app(config)
    app.state.backend_manager = backend_manager
    return app


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
