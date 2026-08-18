from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkItem:
    question_id: str
    domain: str
    question_type: str
    question_text: str
    options_text: str
    reference_answer: str
    image_paths: list[str]
    image_based: bool

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BenchmarkItem":
        return cls(
            question_id=row["question_id"],
            domain=row["domain"],
            question_type=row["question_type"],
            question_text=row["question_text"],
            options_text=row.get("options_text", ""),
            reference_answer=row["reference_answer"],
            image_paths=list(row.get("image_paths", [])),
            image_based=bool(row.get("image_based", False)),
        )


def load_items(path: Path, limit: int | None = None) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(BenchmarkItem.from_dict(json.loads(line)))
            if limit is not None and len(items) >= limit:
                break
    return items


def resolve_image_paths(bench_root: Path, item: BenchmarkItem) -> list[Path]:
    return [(bench_root / rel).resolve() for rel in item.image_paths]

