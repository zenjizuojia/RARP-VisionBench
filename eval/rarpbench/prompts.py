from __future__ import annotations

from pathlib import Path

from .data import BenchmarkItem


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_user_text(item: BenchmarkItem) -> str:
    parts = [
        f"Question ID: {item.question_id}",
        f"Question type: {item.question_type}",
        "",
        item.question_text.strip(),
    ]
    if item.options_text:
        parts.extend(["", "Options:", item.options_text.strip()])
    parts.extend(
        [
            "",
            "The JSON field question_id must exactly match the Question ID above.",
            "For single-choice questions, put one option letter only in the answer field.",
            "For multiple-choice questions, put option letters only in the answer field.",
            "For sequencing questions, put the ordered letters only in the answer field.",
            "For true/false questions, put T or F only in the answer field.",
        ]
    )
    return "\n".join(parts)
