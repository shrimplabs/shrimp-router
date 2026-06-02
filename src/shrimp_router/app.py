"""Shrimp Router — OpenAI-compatible LLM + VLM gateway."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .backends import BackendManager
from .models import ChatCompletionRequest, RouterConfig
from .quota import MinimaxQuotaPoller
from .router import handle_chat

logger = logging.getLogger(__name__)


def load_config(path: str | os.PathLike[str]) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config must be a YAML mapping")
    return loaded


def create_app(config: dict | None = None) -> FastAPI:
    app = FastAPI(title="Shrimp Router", version="0.2.0")
    cfg = config or {}
    app.state.config = cfg

    app.state.quota_pollers: list[MinimaxQuotaPoller] = []

    if cfg.get("backends"):
        router_config = RouterConfig.model_validate(cfg)
        mgr = BackendManager(router_config)
        app.state.backend_manager = mgr

        # Start live quota pollers for any backend that has a MiniMax API key
        for name, backend in router_config.backends.items():
            if backend.auth_env and "minimax" in name.lower():
                api_key = os.environ.get(backend.auth_env)
                if api_key:
                    poller = MinimaxQuotaPoller(api_key, mgr._quota, backend_name=name)
                    poller.start()
                    app.state.quota_pollers.append(poller)
                    logger.info(f"[quota-poller] Live MiniMax quota polling started for '{name}'")
    else:
        app.state.backend_manager = None

    @app.on_event("shutdown")
    async def _shutdown():
        for poller in app.state.quota_pollers:
            poller.stop()
        if app.state.backend_manager:
            await app.state.backend_manager.close()

    @app.get("/health")
    async def health():
        mgr = app.state.backend_manager
        if mgr is None:
            return {"ok": True, "backends": {}}
        results = {}
        for name in mgr.backends:
            results[name] = await mgr.health_check(name)
        quota = mgr.quota_stats()
        return {"ok": True, "backends": results, "quota": quota}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest):
        mgr = request.app.state.backend_manager
        if mgr is None:
            return JSONResponse(status_code=503, content={"error": "No backends configured"})

        result = await handle_chat(request, body)

        if isinstance(result, dict) and "status_code" in result:
            status = result.pop("status_code")
            return JSONResponse(status_code=status, content=result)

        # StreamingResponse passes through directly
        from fastapi.responses import StreamingResponse
        if isinstance(result, StreamingResponse):
            return result

        return result

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = os.environ.get("SHRIMP_ROUTER_CONFIG", "config.yaml")
    config = load_config(config_path)
    listen = config.get("listen", {}) if isinstance(config.get("listen"), dict) else {}
    host = listen.get("host", "127.0.0.1")
    port = int(listen.get("port", 8090))
    logger.info(f"Starting shrimp-router on {host}:{port}")
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
