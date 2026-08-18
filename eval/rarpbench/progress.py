from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable


def load_completed_keys(
    path: Path,
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
    *,
    include_errors: bool = False,
) -> set[tuple[Any, ...]]:
    completed: set[tuple[Any, ...]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") and not include_errors:
                continue
            if row.get("judge_error") and not include_errors:
                continue
            completed.add(key_fn(row))
    return completed


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


class ProgressReporter:
    def __init__(self, *, total: int, label: str, every: int = 1) -> None:
        self.total = max(total, 1)
        self.label = label
        self.every = max(every, 1)
        self.started = time.time()
        self.done = 0
        self.skipped = 0
        self.errors = 0

    def tick(self, *, skipped: bool = False, error: bool = False, detail: str = "") -> None:
        self.done += 1
        if skipped:
            self.skipped += 1
        if error:
            self.errors += 1
        if self.done == 1 or self.done == self.total or self.done % self.every == 0 or error:
            elapsed = time.time() - self.started
            pct = self.done / self.total * 100
            rate = self.done / elapsed if elapsed > 0 else 0
            eta = (self.total - self.done) / rate if rate > 0 else 0
            suffix = f" | {detail}" if detail else ""
            print(
                f"[{self.label}] {self.done}/{self.total} ({pct:.1f}%) "
                f"skipped={self.skipped} errors={self.errors} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s{suffix}",
                flush=True,
            )
