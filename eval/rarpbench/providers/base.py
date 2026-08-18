from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class ModelResponse:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)


class VLMProvider(Protocol):
    provider_name: str

    def generate(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_paths: list[Path],
        temperature: float,
        max_tokens: int,
    ) -> ModelResponse:
        ...

