"""Pydantic models for Shrimp Router."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse


from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Backend / Router config
# ---------------------------------------------------------------------------


class CircuitBreakerConfig(BaseModel):
    """Per-backend circuit breaker configuration."""

    failure_threshold: int = Field(default=3, ge=1)
    cooldown_seconds: int = Field(default=60, ge=1)


class QuotaConfig(BaseModel):
    """Sliding-window quota tracking for a backend."""

    window_seconds: int = 18000  # 5 hours
    max_requests: int = 1000


class BackendConfig(BaseModel):
    """Configuration for a single backend (text or vision)."""


    base_url: str
    models: list[str]
    max_concurrency: int = Field(default=4, ge=1)
    weight: int = Field(default=1, ge=1)  # for weighted round-robin in pools
    tags: list[str] = Field(default_factory=list)  # e.g. ["vision", "text", "vlm"]
    auth_env: str | None = None  # env var name holding the API key
    task_types: list[str] = Field(default_factory=list)  # e.g. ["bug", "polish"]
    quota: QuotaConfig | None = None
    circuit_breaker: CircuitBreakerConfig | None = None
    health_check_path: str = "/health"
    health_check_timeout_seconds: float = Field(default=5.0, gt=0)
    health_check_interval_seconds: float = Field(default=30.0, gt=0)
    format: str = "anthropic"  # wire format this backend expects: "anthropic" or "openai"
    response_format: str = "auto"  # wire format this backend returns: "auto", "openai", or "anthropic"

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be a valid HTTP/HTTPS URL")
        return v

    @field_validator("format")
    @classmethod
    def format_must_be_valid(cls, v: str) -> str:
        allowed = {"openai", "anthropic"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got '{v}'")
        return v

    @field_validator("response_format")
    @classmethod
    def response_format_must_be_valid(cls, v: str) -> str:
        allowed = {"auto", "openai", "anthropic"}
        if v not in allowed:
            raise ValueError(f"response_format must be one of {allowed}, got '{v}'")
        return v

    @property
    def effective_response_format(self) -> str:
        if self.response_format == "auto":
            return self.format
        return self.response_format

    @property
    def api_key(self) -> str | None:
        if self.auth_env:
            return os.environ.get(self.auth_env)
        return None


class RoutingConfig(BaseModel):
    """Task-type to backend preference ordering."""

    # Map task_type -> ordered list of backend names to try
    task_type_backends: dict[str, list[str]] = Field(default_factory=dict)
    # Default backend preference order when no task_type match
    default_backends: list[str] = Field(default_factory=list)
    # Backend names that handle vision requests (image_url in messages)
    vision_backends: list[str] = Field(default_factory=list)
    # Map pipeline phase -> ordered list of backend names (X-Phase header)
    phase_backends: dict[str, list[str]] = Field(default_factory=dict)


class ListenConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8090


class RouterConfig(BaseModel):
    """Top-level router configuration parsed from YAML."""

    listen: ListenConfig = Field(default_factory=ListenConfig)
    request_timeout_seconds: int = 300
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completion types
# ---------------------------------------------------------------------------


class ContentPart(BaseModel):
    """A single part of a multi-modal message (text or image_url)."""


    type: str  # "text" or "image_url"
    text: str | None = None
    image_url: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    """A single message -- content can be a string or list of parts."""

    role: str
    content: str | list[ContentPart] | list[dict[str, Any]]

    def has_images(self) -> bool:
        if isinstance(self.content, str):
            return False
        for part in self.content:
            if isinstance(part, dict):
                if part.get("type") == "image_url":
                    return True
            elif isinstance(part, ContentPart):
                if part.type == "image_url":
                    return True
        return False


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completion request (validated subset)."""


    model: str
    messages: list[ChatMessage]
    stream: bool | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, ge=1)

    def is_vision_request(self) -> bool:
        for msg in self.messages:
            if msg.has_images():
                return True
        return False
