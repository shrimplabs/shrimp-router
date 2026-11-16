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

    assert "/livez" in routes
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


def test_livez_is_cheap_process_probe():
    from fastapi.testclient import TestClient
    from unittest.mock import Mock

    app = create_app()
    manager = Mock()
    manager.health_check.side_effect = AssertionError("/livez must not check backends")
    app.state.backend_manager = manager

    response = TestClient(app).get("/livez")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    manager.health_check.assert_not_called()


def test_backend_model_honors_listed_request_and_defaults_otherwise():
    from types import SimpleNamespace

    from shrimp_router.app import _backend_model

    cfg = SimpleNamespace(models=["MiniMax-M3.1-Flash", "MiniMax-M3"])
    assert _backend_model(cfg, "MiniMax-M3") == "MiniMax-M3"
    assert _backend_model(cfg, "MiniMax-M3.1-Flash") == "MiniMax-M3.1-Flash"
    assert _backend_model(cfg, "gpt-4o") == "MiniMax-M3.1-Flash"
    assert _backend_model(cfg, "") == "MiniMax-M3.1-Flash"
    assert _backend_model(SimpleNamespace(models=[]), "anything") == "anything"
