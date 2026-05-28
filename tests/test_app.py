from shrimp_vision_router.app import create_app, load_config


def test_load_missing_config_returns_empty_dict(tmp_path):
    assert load_config(tmp_path / "missing.yaml") == {}


def test_create_app_has_health_route():
    app = create_app({"backends": {"mini": {"base_url": "http://mini:8080/v1"}}})
    routes = {route.path for route in app.routes}

    assert "/health" in routes

