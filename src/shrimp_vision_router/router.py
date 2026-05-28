"""Proxy endpoint logic for Shrimp Vision Router."""

from __future__ import annotations

import httpx

from .backends import BackendManager
from .models import ChatCompletionRequest, ChatCompletionResponse


class RouteResult:
    """Holds the response and backend name from a routed request."""

    def __init__(self, response: httpx.Response, backend_name: str) -> None:
        self.response = response
        self.backend_name = backend_name


async def route_and_forward(
    manager: BackendManager, request: ChatCompletionRequest
) -> RouteResult:
    """Route the request to an appropriate backend and forward it."""
    matching = manager.backends_for_model(request.model)
    if not matching:
        raise ValueError(f"No backend found for model: {request.model}")

    backend = matching[0]
    # Find the backend name (first key whose value == backend)
    backend_name = next(
        name for name, cfg in manager.backends.items() if cfg is backend
    )

    payload = request.model_dump(mode='json')

    async with (await manager.acquire(backend_name)):
        response = await manager.forward_chat(backend_name, backend, payload)

    return RouteResult(response, backend_name)


async def make_chat_response(route_result: RouteResult, model: str) -> ChatCompletionResponse:
    """Parse backend JSON and replace model name with the requested one."""
    data = route_result.response.json()
    data["model"] = model
    return ChatCompletionResponse.model_validate(data)
