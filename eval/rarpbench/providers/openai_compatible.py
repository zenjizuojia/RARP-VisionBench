from __future__ import annotations

import os
from pathlib import Path

from ..io_utils import image_to_data_url
from .base import ModelResponse
from .http_json import post_json


class OpenAICompatibleProvider:
    """Adapter for OpenAI Chat Completions-compatible vision APIs.

    Works for OpenAI and routers that accept OpenAI-style image_url messages.
    """

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str | None = None,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        if not self.api_key:
            hint = api_key_env if api_key_env and not looks_like_secret(api_key_env) else "<redacted>"
            raise RuntimeError(f"Missing API key. Set api_key or environment variable: {hint}")

    def generate(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_paths: list[Path],
        temperature: float,
        max_tokens: int,
    ) -> ModelResponse:
        content: list[dict] = [{"type": "text", "text": user_text}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        raw = post_json(f"{self.base_url}/chat/completions", headers, payload)
        text = raw["choices"][0]["message"].get("content", "")
        return ModelResponse(text=text, raw=raw)


def looks_like_secret(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("sk-", "sk_", "ak-", "pk-")) or len(value) > 40
