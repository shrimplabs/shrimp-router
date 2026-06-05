"""Shrimp Router -- OpenAI-compatible LLM + VLM gateway."""

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

def _anthropic_to_openai(payload: dict, model: str) -> dict:
    """Translate Anthropic /messages payload to OpenAI /chat/completions payload."""
    messages = []
    system = payload.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system can be a list of content blocks
            text = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
            messages.append({"role": "system", "content": text})

    for msg in payload.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Convert Anthropic content blocks to OpenAI parts
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image":
                        src = block.get("source", {})
                        if src.get("type") == "base64":
                            parts.append({"type": "image_url", "image_url": {
                                "url": f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
                            }})
            messages.append({"role": role, "content": parts if len(parts) > 1 else (parts[0].get("text", "") if parts else "")})

    result: dict = {
        "model": model,
        "messages": messages,
    }
    if payload.get("max_tokens"):
        result["max_tokens"] = payload["max_tokens"]
    if payload.get("temperature") is not None:
        result["temperature"] = payload["temperature"]
    if payload.get("stream"):
        result["stream"] = payload["stream"]
    return result


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
            if name == "minimax" and backend.auth_env and backend.quota:
                api_key = os.environ.get(backend.auth_env)
                if api_key:
                    poller = MinimaxQuotaPoller(api_key, mgr._quota, backend_name=name)
                    poller.start()
                    app.state.quota_pollers.append(poller)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions", name="chat")
    async def chat(request: Request) -> JSONResponse:
        body = await request.json()
        model = body.get("model", "")
        if not model:
            return JSONResponse(status_code=400, content={"error": {"message": "model is required", "type": "invalid_request_error"}})
        try:
            parsed = ChatCompletionRequest.model_validate(body)
        except Exception as e:
            return JSONResponse(status_code=422, content={"error": {"message": str(e), "type": "invalid_request_error"}})

        mgr = app.state.backend_manager
        try:
            result = await handle_chat(mgr, parsed)
        except Exception as e:
            logger.exception("handle_chat failed")
            return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "internal_error"}})
        return JSONResponse(content=result)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shrimp Router")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    application = create_app(cfg)
    uvicorn.run(application, host=args.host, port=args.port)
