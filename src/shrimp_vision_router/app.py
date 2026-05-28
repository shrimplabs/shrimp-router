from __future__ import annotations

import os
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI


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

    @app.get("/health")
    async def health() -> dict:
        backends = app.state.config.get("backends", {})
        return {
            "ok": True,
            "backend_count": len(backends) if isinstance(backends, dict) else 0,
        }

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

