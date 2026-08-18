from __future__ import annotations

import argparse
import json
from pathlib import Path

from rarpbench.response_parse import parse_model_response


def backfill_file(path: Path) -> tuple[int, int]:
    rows = []
    changed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parsed = parse_model_response(row.get("response_text", ""), row.get("question_type", ""))
        old = (
            row.get("parsed_answer", ""),
            row.get("parsed_explanation", ""),
            row.get("parsed_response_valid", ""),
        )
        row["parsed_answer"] = parsed["answer"]
        row["parsed_explanation"] = parsed["explanation"]
        row["parsed_response_valid"] = 1 if parsed["answer"] and parsed.get("json_valid") == "1" else 0
        new = (
            row["parsed_answer"],
            row["parsed_explanation"],
            row["parsed_response_valid"],
        )
        if old != new:
            changed += 1
        rows.append(row)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return len(rows), changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill parsed_answer fields in model-output JSONL files.")
    parser.add_argument("--glob", default="results/model_outputs/*.jsonl")
    parser.add_argument("--benchmark-root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    paths = sorted(root.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")
    for path in paths:
        rows, changed = backfill_file(path)
        print(f"{path}: rows={rows} changed={changed}")


if __name__ == "__main__":
    main()
