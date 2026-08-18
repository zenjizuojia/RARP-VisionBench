from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_answer_key(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?i)^final answer\s*[:：]\s*", "", text)
    text = text.splitlines()[0] if text.splitlines() else text
    text = text.strip().strip(".。;；")
    return text


def normalize_letters(text: str) -> str:
    text = normalize_answer(text).upper()
    letters = re.findall(r"[A-D]", text)
    return "".join(sorted(set(letters)))


def normalize_tf(text: str) -> str:
    text = normalize_answer(text).upper()
    if text in {"T", "TRUE", "YES", "Y"}:
        return "T"
    if text in {"F", "FALSE", "NO", "N"}:
        return "F"
    if "TRUE" in text:
        return "T"
    if "FALSE" in text:
        return "F"
    return text[:1]


def simple_correct(response_text: str, reference: str, question_type: str) -> int | None:
    if question_type in {"single_choice", "multiple_choice"}:
        return int(normalize_letters(response_text) == normalize_letters(reference))
    if question_type == "sequencing":
        response = re.sub(r"[^A-D]", "", normalize_answer(response_text).upper())
        ref = re.sub(r"[^A-D]", "", reference.upper())
        return int(response == ref)
    if question_type == "true_false":
        return int(normalize_tf(response_text) == normalize_tf(reference))
    # Free-text and open-ended items require synonym/rubric adjudication.
    response = normalize_answer(response_text).lower()
    ref = normalize_answer(reference).lower()
    if response == ref:
        return 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score RARP-VisionBench raw model outputs.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("."))
    parser.add_argument("--answers", type=Path, default=Path("data/answer_key.json"))
    parser.add_argument("--outputs", type=Path, default=Path("results/model_outputs/raw_outputs.jsonl"))
    parser.add_argument("--scored-jsonl", type=Path, default=Path("results/scored_outputs/scored_outputs.jsonl"))
    parser.add_argument("--summary-csv", type=Path, default=Path("results/scored_outputs/summary.csv"))
    args = parser.parse_args()

    bench_root = args.benchmark_root.resolve()
    answer_key = load_answer_key(bench_root / args.answers)
    scored_path = bench_root / args.scored_jsonl
    scored_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with (bench_root / args.outputs).open("r", encoding="utf-8") as f, scored_path.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = answer_key[row["question_id"]]
            correct = None if row.get("error") else simple_correct(
                row.get("response_text", ""),
                key["reference_answer"],
                key["question_type"],
            )
            row["reference_answer"] = key["reference_answer"]
            row["auto_correct"] = correct
            row["requires_adjudication"] = correct is None
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups = defaultdict(lambda: {"n_auto": 0, "n_correct": 0, "n_adjudication": 0, "n_error": 0})
    for row in rows:
        group = groups[row["model_alias"]]
        if row.get("error"):
            group["n_error"] += 1
        if row["auto_correct"] is None:
            group["n_adjudication"] += 1
        else:
            group["n_auto"] += 1
            group["n_correct"] += int(row["auto_correct"])

    summary_path = bench_root / args.summary_csv
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_alias", "n_auto_scored", "n_correct", "auto_accuracy", "n_requires_adjudication", "n_error"],
        )
        writer.writeheader()
        for model_alias, stats in sorted(groups.items()):
            acc = stats["n_correct"] / stats["n_auto"] if stats["n_auto"] else ""
            writer.writerow(
                {
                    "model_alias": model_alias,
                    "n_auto_scored": stats["n_auto"],
                    "n_correct": stats["n_correct"],
                    "auto_accuracy": acc,
                    "n_requires_adjudication": stats["n_adjudication"],
                    "n_error": stats["n_error"],
                }
            )


if __name__ == "__main__":
    main()
