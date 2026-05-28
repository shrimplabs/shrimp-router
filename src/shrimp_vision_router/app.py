from __future__ import annotations

import os
from pathlib import Path

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .backends import BackendManager
from .models import ChatCompletionRequest, RouterConfig
from .router import make_chat_response, route_and_forward


def load_config(path: str | os.PathLike[str]) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config must be a mapping")
    return loaded


def create_app(config: dict | None = None) -> FastAPI:
    app = FastAPI(title="Shrimp Vision Router")
    app.state.config = config or {}

    if config:
        router_config = RouterConfig.model_validate(config)
        app.state.backend_manager = BackendManager(
            router_config.backends,
            router_config.request_timeout_seconds,
        )
    else:
        app.state.backend_manager = None

    @app.get("/health")
    async def health() -> dict:
        if app.state.backend_manager is None:
            return {"ok": True, "backend_count": 0}
        results = {}
        for name, backend in app.state.backend_manager.backends.items():
            results[name] = await app.state.backend_manager.health_check(name, backend)
        return results

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest) -> Response:
        manager = request.app.state.backend_manager
        try:
            route_result = await route_and_forward(manager, body)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except httpx.HTTPStatusError as e:
            return JSONResponse(status_code=502, content={"error": "Backend error", "detail": str(e)})

        response = await make_chat_response(route_result, body.model)
        return Response(content=response.model_dump_json(), media_type="application/json")

    return app


def main() -> None:
    config_path = os.environ.get("SHRIMP_VISION_ROUTER_CONFIG", "config.yaml")
    config = load_config(config_path)
    listen = config.get("listen", {}) if isinstance(config.get("listen", {}), dict) else {}
    host = listen.get("host", "127.0.0.1")
    port = int(listen.get("port", 8090))
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()

