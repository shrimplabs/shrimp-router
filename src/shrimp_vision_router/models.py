"""Pydantic models for Shrimp Vision Router."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------------------
# Backend / Router config
# -----------------------------------------------------------------------------


class BackendConfig(BaseModel):
    """Configuration for a single backend worker."""

    base_url: str
    models: list[str]
    max_concurrency: int = Field(ge=1)
    tags: list[str] | None = None
    health_check_path: str | None = Field(default="/health")

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be a valid HTTP/HTTPS URL")
        return v


class ListenConfig(BaseModel):
    """Server listen settings."""

    host: str = "127.0.0.1"
    port: int = 8090


class RouterConfig(BaseModel):
    """Top-level router configuration parsed from YAML."""

    listen: ListenConfig = Field(default_factory=ListenConfig)
    request_timeout_seconds: int = 180
    backends: dict[str, BackendConfig] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# OpenAI-compatible chat completion types
# -----------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions request body."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    extra_body: dict[str, Any] | None = None


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """Token usage statistics for a chat completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI /v1/chat/completions response body."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
