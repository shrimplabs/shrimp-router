from shrimp_router.app import create_app, load_config


def test_load_missing_config_returns_empty_dict(tmp_path):
    assert load_config(tmp_path / "missing.yaml") == {}


def test_create_app_has_public_routes():
    app = create_app({
        "backends": {
            "mini": {
                "base_url": "http://mini:8080/v1",
                "models": ["test-model"],
                "max_concurrency": 1,
            }
        }
    })
    routes = {route.path for route in app.routes}

    assert "/health" in routes
    assert "/metrics" in routes
    assert "/v1/chat/completions" in routes
    assert "/v1/messages" in routes


def test_lifespan_shutdown_stops_pollers_and_backend_manager():
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock, Mock

    app = create_app()
    poller = Mock()
    manager = Mock()
    manager.close = AsyncMock()
    app.state.quota_pollers.append(poller)
    app.state.backend_manager = manager

    with TestClient(app):
        pass

    poller.stop.assert_called_once()
    manager.close.assert_awaited_once()
