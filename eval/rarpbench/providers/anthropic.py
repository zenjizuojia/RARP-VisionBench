from __future__ import annotations

import os
from pathlib import Path

from ..io_utils import image_to_base64
from .base import ModelResponse
from .http_json import post_json


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(self, *, model: str, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing API key environment variable: {api_key_env}")

    def generate(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_paths: list[Path],
        temperature: float,
        max_tokens: int,
    ) -> ModelResponse:
        content: list[dict] = []
        for path in image_paths:
            media_type, data = image_to_base64(path)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }
            )
        content.append({"type": "text", "text": user_text})
        payload = {
            "model": self.model,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        raw = post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload,
        )
        text = "".join(part.get("text", "") for part in raw.get("content", []) if part.get("type") == "text")
        return ModelResponse(text=text, raw=raw)

