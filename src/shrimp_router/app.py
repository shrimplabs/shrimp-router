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
from .router import handle_chat, _stream_response as _stream_anthropic

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

        from fastapi.responses import StreamingResponse
        if isinstance(result, StreamingResponse):
            return result

        return result

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        """Anthropic-format passthrough — forwards raw body to backend /messages endpoint."""
        from fastapi.responses import StreamingResponse as SR
        mgr = request.app.state.backend_manager
        if mgr is None:
            return JSONResponse(status_code=503, content={"error": "No backends configured"})

        body_bytes = await request.body()
        import json
        try:
            payload = json.loads(body_bytes)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

        # Detect vision from Anthropic message format
        is_vision = any(
            isinstance(msg.get("content"), list) and
            any(p.get("type") == "image" for p in msg["content"] if isinstance(p, dict))
            for msg in payload.get("messages", [])
            if isinstance(msg, dict)
        )
        task_type = request.headers.get("X-Task-Type")
        stream = payload.get("stream", False)

        candidates = await mgr.pick_backends(task_type, is_vision)
        if not candidates:
            return JSONResponse(status_code=503, content={"error": "No backends configured"})

        # Forward extra Anthropic headers (anthropic-version, x-api-key, etc.)
        extra_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in ("anthropic-version", "anthropic-beta", "x-api-key")
        }

        last_error = "No backends available"
        for name in candidates:
            if name not in mgr.backends:
                continue
            if not mgr._quota.has_capacity(name):
                logger.info(f"[router] {name} quota exhausted, skipping")
                continue
            try:
                import httpx as _httpx
                cfg = mgr.backends[name]
                headers = {"Content-Type": "application/json", **mgr._auth_headers(name), **extra_headers}
                url = f"{cfg.base_url.rstrip('/')}/messages"
                async with mgr._semaphore(name):
                    mgr._quota.record(name)
                    if stream:
                        req = mgr._client.build_request("POST", url, content=body_bytes, headers=headers)
                        resp = await mgr._client.send(req, stream=True)
                    else:
                        resp = await mgr._client.post(url, content=body_bytes, headers=headers)

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 60))
                    mgr._quota.record_429(name, retry_after)
                    last_error = f"{name}: 429"
                    continue

                if not (200 <= resp.status_code < 300):
                    logger.warning(f"[router] {name} returned {resp.status_code}")
                    last_error = f"{name}: HTTP {resp.status_code}"
                    if 400 <= resp.status_code < 500:
                        return JSONResponse(status_code=resp.status_code, content=resp.json())
                    continue

                logger.info(f"[router] served by {name} (task_type={task_type}, vision={is_vision}, format=anthropic)")

                if stream:
                    return SR(
                        _stream_anthropic(resp, name),
                        media_type="text/event-stream",
                        headers={"X-Backend": name},
                    )

                return JSONResponse(content=resp.json(), headers={"X-Backend": name})

            except _httpx.TimeoutException:
                logger.warning(f"[router] {name} timed out")
                last_error = f"{name}: timeout"
            except Exception as e:
                logger.warning(f"[router] {name} error: {e}")
                last_error = f"{name}: {e}"

        return JSONResponse(status_code=502, content={"error": f"All backends failed: {last_error}"})

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
