from __future__ import annotations

import os
from pathlib import Path

from ..io_utils import image_to_base64
from .base import ModelResponse
from .http_json import post_json


class GeminiProvider:
    provider_name = "gemini"

    def __init__(self, *, model: str, api_key_env: str = "GEMINI_API_KEY") -> None:
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
        parts: list[dict] = [{"text": f"{system_prompt}\n\n{user_text}"}]
        for path in image_paths:
            media_type, data = image_to_base64(path)
            parts.append({"inline_data": {"mime_type": media_type, "data": data}})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        raw = post_json(url, {"Content-Type": "application/json"}, payload)
        text = ""
        candidates = raw.get("candidates", [])
        if candidates:
            text = "".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", []))
        return ModelResponse(text=text, raw=raw)

