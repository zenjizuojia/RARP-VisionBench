from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model_alias"),
        row.get("repeat_idx"),
        row.get("question_id"),
        row.get("prompt_file"),
    )


def is_success(row: dict[str, Any]) -> bool:
    return not row.get("error") and bool(row.get("parsed_response_valid", 0))


def choose_row(old: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    if old is None:
        return new
    if is_success(new) and not is_success(old):
        return new
    if is_success(new) == is_success(old):
        return new
    return old


def compact_file(path: Path, *, dry_run: bool) -> tuple[int, int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = row_key(row)
        if key not in latest:
            order.append(key)
            latest[key] = row
        else:
            latest[key] = choose_row(latest[key], row)
    compacted = [latest[key] for key in order]
    if not dry_run and len(compacted) != len(rows):
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in compacted) + "\n", encoding="utf-8")
    errors = sum(1 for row in compacted if row.get("error"))
    return len(rows), len(compacted), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact model-output JSONL files to one best row per question.")
    parser.add_argument("--glob", default="results/model_outputs/*.jsonl")
    parser.add_argument("--benchmark-root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    paths = sorted(root.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")
    for path in paths:
        before, after, errors = compact_file(path, dry_run=args.dry_run)
        print(f"{path}: rows {before} -> {after}; remaining_errors={errors}")


if __name__ == "__main__":
    main()
