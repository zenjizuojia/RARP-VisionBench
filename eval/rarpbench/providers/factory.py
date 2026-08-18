from __future__ import annotations

import os
from typing import Any

from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider


def get_base_url(spec: dict[str, Any], default: str) -> str:
    env_name = spec.get("base_url_env")
    if env_name:
        if str(env_name).startswith(("http://", "https://")):
            return str(env_name)
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(f"Missing base URL environment variable: {env_name}")
        return value
    return spec.get("base_url", default)


def get_api_key_args(spec: dict[str, Any], default_env: str) -> dict[str, str | None]:
    api_key = spec.get("api_key")
    api_key_env = spec.get("api_key_env", default_env)
    if api_key_env and looks_like_secret(str(api_key_env)) and not api_key:
        api_key = str(api_key_env)
        api_key_env = None
    return {"api_key": api_key, "api_key_env": api_key_env}


def looks_like_secret(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("sk-", "sk_", "ak-", "pk-")) or len(value) > 40


def make_provider(spec: dict[str, Any]):
    kind = spec["provider"]
    if kind == "openai":
        return OpenAICompatibleProvider(
            model=spec["model"],
            **get_api_key_args(spec, "OPENAI_API_KEY"),
            base_url=get_base_url(spec, "https://api.openai.com/v1"),
        )
    if kind == "openrouter":
        headers = {}
        if spec.get("http_referer"):
            headers["HTTP-Referer"] = spec["http_referer"]
        if spec.get("x_title"):
            headers["X-Title"] = spec["x_title"]
        return OpenAICompatibleProvider(
            model=spec["model"],
            **get_api_key_args(spec, "OPENROUTER_API_KEY"),
            base_url=get_base_url(spec, "https://openrouter.ai/api/v1"),
            extra_headers=headers,
        )
    if kind == "openai_compatible":
        return OpenAICompatibleProvider(
            model=spec["model"],
            **get_api_key_args(spec, "OPENAI_API_KEY"),
            base_url=get_base_url(spec, "https://api.openai.com/v1"),
            extra_headers=spec.get("extra_headers", {}),
        )
    if kind == "anthropic":
        return AnthropicProvider(model=spec["model"], api_key_env=spec.get("api_key_env", "ANTHROPIC_API_KEY"))
    if kind == "gemini":
        return GeminiProvider(model=spec["model"], api_key_env=spec.get("api_key_env", "GEMINI_API_KEY"))
    raise ValueError(f"Unknown provider kind: {kind}")
