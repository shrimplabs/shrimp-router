"""Shrimp Router \u2014 OpenAI-compatible LLM + VLM gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
import yaml
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .backends import BackendManager
from .format_normalizer import openai_to_anthropic_response
from .models import ChatCompletionRequest, RouterConfig
from .quota import MinimaxQuotaPoller
from .router import handle_chat, _stream_response as _stream_anthropic

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        for poller in app.state.quota_pollers:
            poller.stop()
        if app.state.backend_manager:
            await app.state.backend_manager.close()


async def _release_inflight(body_iter, mgr, name: str):
    """Pass a streamed body through, freeing the backend's in-flight slot when
    it ends (finished, failed, or the client went away)."""
    try:
        async for chunk in body_iter:
            yield chunk
    finally:
        mgr._inflight[name] -= 1


def _backend_model(cfg, requested: str) -> str:
    """Model to send to a backend: the client's model if the backend lists it,
    otherwise the backend's default (first listed) model."""
    if not cfg.models:
        return requested
    return requested if requested in cfg.models else cfg.models[0]


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
        # Usage arrives in a final chunk only when asked for; the Anthropic
        # stream translation reports it as output_tokens.
        result["stream_options"] = {"include_usage": True}
    return result


def _session_id(request, payload: dict) -> str:
    """Stable per-conversation id: the caller's X-Session-Id (the swarm sends its
    task id), else a hash of the system prompt and first message."""
    explicit = request.headers.get("X-Session-Id")
    if explicit:
        return explicit
    import hashlib
    msgs = payload.get("messages") or []
    seed = json.dumps([payload.get("system"), msgs[0] if msgs else None], sort_keys=True, default=str)
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


def _client_headers(cfg, request, payload: dict) -> dict:
    headers = {}
    if cfg.session_header:
        headers[cfg.session_header] = _session_id(request, payload)
    if cfg.user_agent:
        headers["User-Agent"] = cfg.user_agent
    return headers


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
    app = FastAPI(title="Shrimp Router", version="0.2.0", lifespan=_lifespan)
    cfg = config or {}
    app.state.config = cfg

    app.state.quota_pollers: list[MinimaxQuotaPoller] = []

    if cfg.get("backends"):
        router_config = RouterConfig.model_validate(cfg)
        mgr = BackendManager(router_config)
        app.state.backend_manager = mgr

        # Live MiniMax quota polling: at cutover_percent every backend on the
        # same key (minimax, minimax-fast, ...) is skipped until the reset.
        mm = router_config.backends.get("minimax")
        if mm and mm.auth_env and mm.quota:
            api_key = os.environ.get(mm.auth_env)
            if api_key:
                names = [n for n, b in router_config.backends.items() if b.auth_env == mm.auth_env]
                for n in names:
                    if mgr._quota.get(n) is None:
                        mgr._quota.register(n, mm.quota.window_seconds, mm.quota.max_requests)
                poller = MinimaxQuotaPoller(api_key, mgr._quota, backend_name="minimax",
                                            cutover_percent=mm.quota.cutover_percent, backend_names=names)
                poller.start()
                app.state.quota_pollers.append(poller)
                logger.info(f"[quota-poller] Live MiniMax quota polling for {names} "
                            f"(cutover {mm.quota.cutover_percent}%)")
    else:
        app.state.backend_manager = None

    @app.get("/livez")
    async def livez():
        """Cheap process liveness probe.

        Unlike /health, this deliberately does not check remote backends. It is
        safe for watchdogs/process supervisors that only need to know whether
        shrimp-router itself is accepting requests.
        """
        return {"ok": True}

    @app.get("/health")
    async def health():
        mgr = app.state.backend_manager
        if mgr is None:
            return {"ok": True, "backends": {}}
        results = {}
        for name in mgr.backends:
            results[name] = await mgr.health_check(name)
        quota = mgr.quota_stats()
        return {
            "ok": True,
            "backends": results,
            "quota": quota,
            "circuit_breakers": mgr.circuit_breaker_stats(),
        }

    @app.get("/metrics")
    async def metrics():
        mgr = app.state.backend_manager
        if mgr is None:
            return {"backends": {}}
        return {"backends": mgr.metrics()}


    async def _stream_responses_api(resp: "httpx.Response", client_model: str):
        """Translate chat.completions SSE stream → Responses API SSE format.

        Handles two output shapes from Athena:
          1. Text response: response.created → output_item.added (message) →
             content_part.added → text.delta (N) → text.done →
             output_item.done → response.completed
          2. Tool call: response.created → output_item.added (function_call) →
             function_call_arguments.delta (N) → output_item.done →
             response.completed
        We don't know which until the first delta arrives, so we buffer the
        first meaningful chunk before emitting the header events.
        """
        import uuid
        resp_id = f"resp_{uuid.uuid4().hex[:20]}"

        def _ev(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        # --- collect all chunks first so we can detect tool vs text ---
        # (Athena responses are fast; buffering is fine for local traffic)
        chunks = []
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunks.append(json.loads(data_str))
            except Exception:
                continue

        # Detect if this response is a tool call by inspecting finish_reason
        # or presence of tool_calls in any delta.
        finish_reason = None
        for chunk in chunks:
            fr = chunk.get("choices", [{}])[0].get("finish_reason")
            if fr:
                finish_reason = fr
        is_tool_call = finish_reason == "tool_calls" or any(
            chunk.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
            for chunk in chunks
        )

        yield _ev("response.created", {
            "type": "response.created",
            "response": {"id": resp_id, "object": "realtime.response", "status": "in_progress",
                         "output": [], "model": client_model}
        })

        if is_tool_call:
            # Collect all tool call deltas by index
            # delta.tool_calls is a list of {index, id, type, function:{name, arguments}}
            # where name arrives in the first chunk and arguments accumulate as deltas.
            tool_slots: dict[int, dict] = {}  # index → {id, name, args_chunks}
            for chunk in chunks:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    if idx not in tool_slots:
                        tool_slots[idx] = {"id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                           "name": "", "args_chunks": []}
                    if tc.get("function", {}).get("name"):
                        tool_slots[idx]["name"] = tc["function"]["name"]
                    if tc.get("function", {}).get("arguments"):
                        tool_slots[idx]["args_chunks"].append(tc["function"]["arguments"])

            output_items = []
            for out_idx, (_, slot) in enumerate(sorted(tool_slots.items())):
                call_id = slot["id"]
                fn_name = slot["name"]
                fn_item_id = f"fc_{uuid.uuid4().hex[:20]}"

                yield _ev("response.output_item.added", {
                    "type": "response.output_item.added", "output_index": out_idx,
                    "item": {"id": fn_item_id, "type": "function_call",
                             "call_id": call_id, "name": fn_name,
                             "arguments": "", "status": "in_progress"}
                })
                for arg_chunk in slot["args_chunks"]:
                    yield _ev("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "output_index": out_idx, "item_id": fn_item_id,
                        "call_id": call_id, "delta": arg_chunk
                    })
                full_args = "".join(slot["args_chunks"])
                yield _ev("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "output_index": out_idx, "item_id": fn_item_id,
                    "call_id": call_id, "arguments": full_args
                })
                yield _ev("response.output_item.done", {
                    "type": "response.output_item.done", "output_index": out_idx,
                    "item": {"id": fn_item_id, "type": "function_call",
                             "call_id": call_id, "name": fn_name,
                             "arguments": full_args, "status": "completed"}
                })
                output_items.append({"type": "function_call", "id": fn_item_id,
                                     "call_id": call_id, "name": fn_name,
                                     "arguments": full_args})

            yield _ev("response.completed", {
                "type": "response.completed",
                "response": {"id": resp_id, "object": "realtime.response", "status": "completed",
                             "output": output_items, "model": client_model,
                             "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
            })

        else:
            # Text response path
            msg_item_id = f"msg_{uuid.uuid4().hex[:20]}"
            yield _ev("response.output_item.added", {
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": msg_item_id, "type": "message",
                         "role": "assistant", "content": [], "status": "in_progress"}
            })
            yield _ev("response.content_part.added", {
                "type": "response.content_part.added", "item_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            })

            full_text = []
            for chunk in chunks:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content") or ""
                if text:
                    full_text.append(text)
                    yield _ev("response.text.delta", {
                        "type": "response.text.delta", "item_index": 0,
                        "content_index": 0, "delta": text
                    })

            assembled = "".join(full_text)
            yield _ev("response.text.done", {
                "type": "response.text.done", "item_index": 0, "content_index": 0,
                "text": assembled
            })
            yield _ev("response.output_item.done", {
                "type": "response.output_item.done", "output_index": 0,
                "item": {"id": msg_item_id, "type": "message",
                         "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": assembled}]}
            })
            yield _ev("response.completed", {
                "type": "response.completed",
                "response": {"id": resp_id, "object": "realtime.response", "status": "completed",
                             "output": [{"type": "message", "role": "assistant",
                                         "content": [{"type": "output_text", "text": assembled}]}],
                             "model": client_model,
                             "usage": {"input_tokens": 0, "output_tokens": len(full_text),
                                       "total_tokens": len(full_text)}}
            })

    @app.get("/v1/models")
    async def list_models(request: Request):
        """Stub /v1/models — Codex CLI calls this on startup for version check.
        Returns a minimal OpenAI-compatible model list so Codex doesn't abort.
        """
        return JSONResponse({
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "openai"},
                {"id": "o3", "object": "model", "created": 1715367049, "owned_by": "openai"},
                {"id": "o4-mini", "object": "model", "created": 1715367049, "owned_by": "openai"},
            ]
        })

    @app.post("/v1/responses")
    async def openai_responses(request: Request):
        """Translate OpenAI Responses API (/v1/responses) to /v1/chat/completions.

        Codex CLI uses the newer Responses API. We convert it to chat completions
        format and forward through the normal routing path.
        """
        mgr = request.app.state.backend_manager
        if mgr is None:
            return JSONResponse(status_code=503, content={"error": "No backends configured"})

        body_bytes = await request.body()
        try:
            payload = json.loads(body_bytes)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

        logger.info(f"[responses] payload keys={list(payload.keys())} model={payload.get('model')}")
        # Dump full payload to file for debugging
        import pathlib
        pathlib.Path("/tmp/codex-last-payload.json").write_text(json.dumps(payload, indent=2))

        # Translate Responses API to chat completions format
        # Responses API: {model, input: str|list, instructions: str, stream: bool, ...}
        # Chat completions: {model, messages: [...], stream: bool, ...}
        messages = []
        instructions = payload.get("instructions") or payload.get("system")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        input_data = payload.get("input", "")
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            # Can be a list of message objects or content parts
            for item in input_data:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    messages.append({"role": role, "content": content})
                elif isinstance(item, str):
                    messages.append({"role": "user", "content": item})

        chat_payload = {
            "model": payload.get("model", "gpt-4o"),
            "messages": messages,
            "stream": payload.get("stream", False),
        }
        # Forward optional params
        for key in ("temperature", "max_tokens", "max_output_tokens", "top_p", "tools", "tool_choice"):
            if key in payload:
                # Responses API uses max_output_tokens, chat uses max_tokens
                dest_key = "max_tokens" if key == "max_output_tokens" else key
                chat_payload[dest_key] = payload[key]

        # Route Codex requests directly — openai passthrough (OAuth token forwarded)
        # with Athena for read-only tool calls. This is isolated from the swarm
        # /v1/messages path so global default_backends is not affected.
        incoming_auth = request.headers.get("Authorization")
        from .router import _try_backend
        from .tool_classifier import classify_last_tool

        tool_class = classify_last_tool(chat_payload.get("messages", []))
        # read = confirmed read-only tool loop → athena (free local)
        # unknown = first turn / no prior tools → athena (cheap, just context loading)
        # write = write tool used last → openai (needs strong model)
        backend_name = "openai" if tool_class == "write" else "athena"
        # Fall back to openai if athena is unavailable
        if backend_name == "athena" and "athena" not in mgr.backends:
            backend_name = "openai"

        client_model = payload.get("model", "gpt-4o")
        # When routing to Athena, substitute the actual loaded model name.
        # Athena rejects requests for model names it doesn't have (e.g. "gpt-4o").
        def _pick_model(bname: str) -> str:
            if bname == "athena" and "athena" in mgr.backends:
                athena_models = mgr.backends["athena"].models
                return athena_models[0] if athena_models else client_model
            return client_model

        athena_model = athena_model_name or _pick_model("athena")
        # Athena needs a real local model name; OpenAI path keeps the original
        # Codex model (gpt-5.5 etc.) — headroom forwards the OAuth token so
        # Codex-internal models are available on that path.
        _ATHENA_MODEL_MAP = {
            "gpt-5.5": "gpt-4o",
            "gpt-5": "gpt-4o",
        }
        athena_model_name = _ATHENA_MODEL_MAP.get(client_model, "gpt-4o")
        oai_model = client_model  # pass through as-is for headroom → OpenAI

        # Strip Responses API-only fields that Athena/llama.cpp rejects
        _ATHENA_ALLOWED = {"model", "messages", "tools", "tool_choice",
                           "max_tokens", "temperature", "top_p", "stream", "stop", "n",
                           "presence_penalty", "frequency_penalty", "seed", "logprobs"}
        athena_payload_base = {k: v for k, v in chat_payload.items() if k in _ATHENA_ALLOWED}

        # Normalize messages for Athena/llama.cpp compatibility:
        # 1. Flatten content arrays to plain strings (Athena rejects content[].type)
        # 2. Remap unknown roles: "developer" → "system"
        # 3. Collapse consecutive same-role messages (llama.cpp rejects 2+ assistant at end)
        if "messages" in athena_payload_base:
            fixed_messages = []
            for msg in athena_payload_base["messages"]:
                role = msg.get("role", "user")
                if role == "developer":
                    role = "system"
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                # tool role: pass through as-is (llama.cpp supports it), just fix content
                # assistant messages with tool_calls: preserve tool_calls field
                if role == "tool":
                    fixed_messages.append({"role": "tool",
                                           "tool_call_id": msg.get("tool_call_id", ""),
                                           "content": content})
                    continue
                if role == "assistant" and msg.get("tool_calls"):
                    # preserve tool_calls for context; don't merge with adjacent
                    fixed_messages.append({"role": "assistant",
                                           "content": content or None,
                                           "tool_calls": msg["tool_calls"]})
                    continue
                # Merge consecutive same-role messages (skip tool/assistant-with-calls)
                if fixed_messages and fixed_messages[-1]["role"] == role and role not in ("tool",):
                    fixed_messages[-1]["content"] += "\n" + content
                else:
                    fixed_messages.append({**msg, "role": role, "content": content})
            athena_payload_base["messages"] = fixed_messages

        # Normalize tools for llama.cpp:
        # 1. Only keep type=function tools (drop custom/computer_use/etc.)
        # 2. Rewrite flat format {type,name,description,parameters} → nested {type,function:{...}}
        if "tools" in athena_payload_base:
            fixed_tools = []
            for t in athena_payload_base["tools"]:
                if t.get("type") != "function":
                    continue  # llama.cpp only supports function tools
                if "function" not in t:
                    # flat format → nested
                    fixed_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        }
                    })
                else:
                    fixed_tools.append(t)
            if fixed_tools:
                athena_payload_base["tools"] = fixed_tools
            else:
                athena_payload_base.pop("tools", None)
                athena_payload_base.pop("tool_choice", None)

        logger.info(f"[responses] routing to {backend_name}, keys after strip: {list(athena_payload_base.keys())}")
        stream = payload.get("stream", False)
        if backend_name == "athena":
            athena_payload = {**athena_payload_base, "model": athena_model}
            import pathlib
            pathlib.Path("/tmp/codex-athena-payload.json").write_text(json.dumps(athena_payload, indent=2))
            resp, err = await _try_backend(mgr, "athena", athena_payload, stream=stream, model=athena_model, incoming_auth=incoming_auth)
            if resp is None:
                # Athena failed, fall back to openai with original model
                resp, err = await _try_backend(mgr, "openai", chat_payload, stream=stream, model=oai_model, incoming_auth=incoming_auth)
        else:
            resp, err = await _try_backend(mgr, "openai", chat_payload, stream=stream, model=oai_model, incoming_auth=incoming_auth)
        if resp is None:
            return JSONResponse(status_code=502, content={"error": err or "All backends failed"})

        if payload.get("stream"):
            return StreamingResponse(
                _stream_responses_api(resp, payload.get("model", "gpt-4o")),
                media_type="text/event-stream"
            )

        try:
            result = resp.json()
        except Exception:
            return JSONResponse(status_code=502, content={"error": "Invalid JSON from backend"})

        # Wrap response back in Responses API shape
        if isinstance(result, StreamingResponse):
            return result  # streaming passthrough

        if isinstance(result, dict):
            status = result.pop("status_code", 200)
            if status != 200:
                return JSONResponse(status_code=status, content=result)
            # Convert chat completions response to responses API format
            choice = (result.get("choices") or [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content", "")
            responses_result = {
                "id": result.get("id", "resp_shrimp"),
                "object": "response",
                "model": result.get("model", chat_payload["model"]),
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}],
                "usage": result.get("usage", {}),
            }
            # Include tool calls if present
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                responses_result["output"].extend([
                    {"type": "function_call", "name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "call_id": tc.get("id", "")}
                    for tc in tool_calls if isinstance(tc, dict)
                ])
            return JSONResponse(content=responses_result)

        return result

    @app.websocket("/v1/responses")
    async def openai_responses_ws(websocket: WebSocket):
        """WebSocket handler — immediately closes so Codex falls back to HTTPS POST.

        The WS payload normalization is incomplete; POST path works correctly.
        Accept then close with 1001 (going away) triggers Codex's HTTPS fallback.
        """
        await websocket.accept()
        await websocket.close(code=1001)
        return
        # --- dead code below, kept for reference ---
        await websocket.accept()
        try:
            while True:
                try:
                    raw = await websocket.receive_text()
                except WebSocketDisconnect:
                    break
                try:
                    payload = json.loads(raw)
                except Exception:
                    await websocket.send_text(json.dumps({"error": "invalid json"}))
                    continue

                # Reuse the same POST handler logic by constructing a mock scope
                # — simpler: just forward directly to the same translation pipeline
                from .tool_classifier import classify_last_tool

                client_model = payload.get("model", "gpt-4o")
                # Translate input → messages (same as POST handler)
                chat_payload = dict(payload)
                input_items = chat_payload.pop("input", None)
                if input_items is not None:
                    messages = []
                    sys_inst = chat_payload.pop("instructions", None)
                    if sys_inst:
                        messages.append({"role": "system", "content": sys_inst})
                    if isinstance(input_items, str):
                        messages.append({"role": "user", "content": input_items})
                    elif isinstance(input_items, list):
                        for item in input_items:
                            if isinstance(item, dict):
                                itype = item.get("type", "")
                                if itype == "message":
                                    messages.append({"role": item.get("role", "user"),
                                                     "content": item.get("content", "")})
                                elif itype == "function_call_output":
                                    messages.append({"role": "tool",
                                                     "tool_call_id": item.get("call_id", ""),
                                                     "content": str(item.get("output", ""))})
                    chat_payload["messages"] = messages

                tool_class = classify_last_tool(chat_payload.get("messages", []))
                backend_name = "openai" if tool_class == "write" else "athena"

                mgr = websocket.app.state.backend_manager
                _ATHENA_ALLOWED = {"model", "messages", "tools", "tool_choice",
                                   "max_tokens", "temperature", "top_p", "stream",
                                   "stop", "n", "presence_penalty", "frequency_penalty",
                                   "seed", "logprobs"}
                athena_payload = {k: v for k, v in chat_payload.items() if k in _ATHENA_ALLOWED}
                athena_payload["stream"] = True
                # The athena backend's configured model, not a hardcoded name:
                # athena swaps models (laguna -> qwen-flash-next) and this went stale.
                _athena_cfg = mgr.backends.get("athena") if mgr else None
                athena_payload["model"] = (_athena_cfg.models[0] if _athena_cfg and _athena_cfg.models
                                           else client_model)

                from .router import _try_backend
                incoming_auth = None  # WS doesn't have headers easily; OAuth not needed for athena
                fwd_payload = athena_payload if backend_name == "athena" else chat_payload
                resp, err = await _try_backend(mgr, backend_name, fwd_payload,
                                               stream=True, model=fwd_payload.get("model", client_model),
                                               incoming_auth=incoming_auth)
                if err or resp is None:
                    await websocket.send_text(json.dumps({"type": "error", "error": err or "no response"}))
                    continue

                async for event_str in _stream_responses_api(resp, client_model):
                    await websocket.send_text(event_str.strip())

        except Exception as e:
            logger.exception("WebSocket /v1/responses error: %s", e)
            try:
                await websocket.close(code=1011)
            except Exception:
                pass

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest):
        mgr = request.app.state.backend_manager
        if mgr is None:
            return JSONResponse(status_code=503, content={"error": "No backends configured"})

        result = await handle_chat(request, body)

        if isinstance(result, dict) and "status_code" in result:
            status = result.pop("status_code")
            return JSONResponse(status_code=status, content=result)

        if isinstance(result, StreamingResponse):
            return result

        return result

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        """Anthropic-format passthrough \u2014 forwards raw body to backend /messages endpoint."""
        mgr = request.app.state.backend_manager
        if mgr is None:
            return JSONResponse(status_code=503, content={"error": "No backends configured"})

        body_bytes = await request.body()
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
        phase = request.headers.get("X-Phase")
        stream = payload.get("stream", False)

        candidates = await mgr.pick_backends(task_type, is_vision, phase=phase)
        if not candidates:
            return JSONResponse(status_code=503, content={"error": "No backends configured"})


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
            cap = mgr.backends[name].max_inflight
            if cap and mgr._inflight.get(name, 0) >= cap:
                logger.info(f"[router] {name} at max_inflight {cap}, skipping")
                last_error = f"{name}: at max_inflight {cap}"
                continue
            mgr._inflight[name] = mgr._inflight.get(name, 0) + 1
            handed_off = False  # a streamed response releases the slot when it ends
            try:
                import httpx as _httpx
                cfg = mgr.backends[name]
                backend_format = cfg.format  # "anthropic" or "openai"

                if backend_format == "openai":
                    # Translate Anthropic \u2192 OpenAI format
                    fwd_payload = _anthropic_to_openai(payload, _backend_model(cfg, payload.get("model", "")))
                    fwd_bytes = json.dumps(fwd_payload).encode()
                    fwd_headers = {"Content-Type": "application/json", **mgr._auth_headers(name),
                                   **_client_headers(cfg, request, payload)}
                    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
                else:
                    # Passthrough Anthropic \u2192 Anthropic, rewrite model to backend's model
                    if cfg.models:
                        fwd_payload = {**payload, "model": _backend_model(cfg, payload.get("model", ""))}
                        fwd_bytes = json.dumps(fwd_payload).encode()
                    else:
                        fwd_bytes = body_bytes
                    fwd_headers = {"Content-Type": "application/json", **mgr._auth_headers(name), **extra_headers,
                                   **_client_headers(cfg, request, payload)}
                    url = f"{cfg.base_url.rstrip('/')}/messages"

                # Acquire semaphore only long enough to send the request, then
                # release immediately — prevents slow Kimi/DeepSeek responses
                # from holding slots and starving the uvicorn event loop.
                async with mgr._semaphore(name):
                    mgr._quota.record(name)
                    req = mgr._client.build_request("POST", url, content=fwd_bytes, headers=fwd_headers)
                    resp = await mgr._client.send(req, stream=True)
                # Semaphore released — slot free for next request

                if not stream:
                    try:
                        await asyncio.wait_for(resp.aread(), timeout=mgr._timeout)
                    except asyncio.TimeoutError:
                        await resp.aclose()
                        last_error = f"{name}: body read timed out"
                        continue

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 60))
                    mgr._quota.record_429(name, retry_after)
                    last_error = f"{name}: 429"
                    # A skipped stream still holds its pooled connection until
                    # closed; enough of them and every request waits on the pool.
                    await resp.aclose()
                    continue

                if not (200 <= resp.status_code < 300):
                    logger.warning(f"[router] {name} returned {resp.status_code}")
                    last_error = f"{name}: HTTP {resp.status_code}"
                    if 400 <= resp.status_code < 500:
                        # Must read body before calling .json() — response was opened
                        # with stream=True so the body isn't buffered yet.
                        await resp.aread()
                        return JSONResponse(status_code=resp.status_code, content=resp.json())
                    await resp.aclose()
                    continue

                logger.info(f"[router] served by {name} (task_type={task_type}, phase={phase}, vision={is_vision}, format={cfg.format})")
                if stream:
                    if backend_format == "openai":
                        from .format_normalizer import openai_stream_to_anthropic
                        body_iter = openai_stream_to_anthropic(resp, payload.get("model", cfg.models[0] if cfg.models else ""))
                    else:
                        body_iter = _stream_anthropic(resp, name)
                    body_iter = _release_inflight(body_iter, mgr, name)
                    handed_off = True
                    return StreamingResponse(
                        body_iter,
                        media_type="text/event-stream",
                        headers={"X-Backend": name},
                    )

                resp_json = resp.json()
                if cfg.effective_response_format == "openai":
                    original_model = payload.get("model", cfg.models[0] if cfg.models else "")
                    resp_json = openai_to_anthropic_response(resp_json, original_model)
                return JSONResponse(content=resp_json, headers={"X-Backend": name})

            except _httpx.TimeoutException:
                logger.warning(f"[router] {name} timed out")
                last_error = f"{name}: timeout"
                cb = mgr.circuit_breaker(name)
                if cb is not None:
                    cb.record_failure()
            except Exception as e:
                logger.warning(f"[router] {name} error: {e}")
                last_error = f"{name}: {e}"
                cb = mgr.circuit_breaker(name)
                if cb is not None:
                    cb.record_failure()
            finally:
                if not handed_off:
                    mgr._inflight[name] -= 1

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
