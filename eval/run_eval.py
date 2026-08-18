from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rarpbench.data import load_items, resolve_image_paths
from rarpbench.prompts import build_user_text, load_prompt
from rarpbench.progress import ProgressReporter, load_completed_keys
from rarpbench.response_parse import parse_model_response
from rarpbench.providers.factory import make_provider


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model_alias"),
        row.get("repeat_idx"),
        row.get("question_id"),
        row.get("prompt_file"),
    )


def default_output_path(out_dir: Path, model_alias: str, repeat_idx: int, prompt_path: Path) -> Path:
    prompt_name = prompt_path.stem
    return out_dir / f"{model_alias}_repeat{repeat_idx:02d}_{prompt_name}.jsonl"


def summarize_raw_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep enough API metadata for debugging without storing the full response."""
    choices = raw.get("choices") or []
    if not choices:
        return {"has_choices": False}
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    return {
        "has_choices": True,
        "finish_reason": choice.get("finish_reason"),
        "message_keys": sorted(message.keys()),
        "content_type": type(content).__name__,
        "content_length": len(content) if isinstance(content, str) else None,
    }


def is_retryable_error(exc: Exception) -> bool:
    text = repr(exc)
    return any(
        marker in text
        for marker in (
            "URLError",
            "ConnectionResetError",
            "timed out",
            "TimeoutError",
            "HTTP 429",
            "HTTP 500",
            "HTTP 502",
            "HTTP 503",
            "HTTP 504",
            "HTTP 525",
        )
    )


def is_retryable_empty_response(metadata: dict[str, Any]) -> bool:
    return metadata.get("finish_reason") in {"length", "content_filter", None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RARP-VisionBench VLM evaluation.")
    parser.add_argument("--config", type=Path, default=Path("eval/config/models.example.json"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("."))
    parser.add_argument("--questions", type=Path, default=Path("data/questions_full.jsonl"))
    parser.add_argument("--prompt", type=Path, default=Path("prompts/direct_answer.txt"))
    parser.add_argument("--out", type=Path, default=None, help="Optional single JSONL output file. If omitted, output is split by model and repeat.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/model_outputs"), help="Directory for split model outputs.")
    parser.add_argument("--models", nargs="*", help="Optional subset of model aliases from the config.")
    parser.add_argument("--question-ids", nargs="*", help="Optional subset of question IDs to run.")
    parser.add_argument("--rerun-selected", action="store_true", help="Rerun selected question IDs even if completed, appending new rows without clearing the output file.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=2, help="Retry transient API/network errors N times before recording failure.")
    parser.add_argument("--retry-sleep", type=float, default=5.0, help="Seconds to wait between retries.")
    parser.add_argument("--dry-run", action="store_true", help="Build requests but do not call APIs.")
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs and rerun all selected items.")
    parser.add_argument("--skip-errors", action="store_true", help="Treat existing error rows as completed when resuming.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N planned items.")
    args = parser.parse_args()
    if args.rerun_selected and not args.question_ids:
        raise SystemExit("--rerun-selected must be used together with --question-ids.")

    bench_root = args.benchmark_root.resolve()
    load_dotenv(bench_root / ".env")
    config = load_config(bench_root / args.config)
    model_specs = config["models"]
    if args.models:
        wanted = set(args.models)
        model_specs = [m for m in model_specs if m["alias"] in wanted]
    if not model_specs:
        raise SystemExit("No models selected.")

    items = load_items(bench_root / args.questions, limit=args.limit)
    if args.question_ids:
        wanted_question_ids = set(args.question_ids)
        items = [item for item in items if item.question_id in wanted_question_ids]
        missing_question_ids = sorted(wanted_question_ids - {item.question_id for item in items})
        if missing_question_ids:
            raise SystemExit(f"Question IDs not found in {args.questions}: {', '.join(missing_question_ids)}")
    system_prompt = load_prompt(bench_root / args.prompt)

    def get_out_path(model_alias: str, repeat_idx: int) -> Path:
        if args.out is not None:
            return bench_root / args.out
        return bench_root / default_output_path(args.out_dir, model_alias, repeat_idx, args.prompt)

    planned_paths = {
        get_out_path(spec["alias"], repeat_idx)
        for spec in model_specs
        for repeat_idx in range(1, args.repeats + 1)
    }
    if args.force:
        for path in planned_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
    completed_by_path: dict[Path, set[tuple[Any, ...]]] = {}
    if not args.force:
        completed_by_path = {
            path: load_completed_keys(path, output_key, include_errors=args.skip_errors)
            for path in planned_paths
        }

    providers = {}
    if not args.dry_run:
        providers = {spec["alias"]: make_provider(spec) for spec in model_specs}

    planned_total = len(model_specs) * args.repeats * len(items)
    reporter = ProgressReporter(total=planned_total, label="run_eval", every=args.progress_every)
    print(
        f"[run_eval] selected_models={len(model_specs)} items={len(items)} repeats={args.repeats} "
        f"planned={planned_total} resume_skippable={sum(len(v) for v in completed_by_path.values())} "
        f"out={'single:' + str(bench_root / args.out) if args.out is not None else 'split:' + str(bench_root / args.out_dir)}",
        flush=True,
    )

    for spec in model_specs:
        alias = spec["alias"]
        provider = providers.get(alias)
        for repeat_idx in range(1, args.repeats + 1):
            out_path = get_out_path(alias, repeat_idx)
            completed = completed_by_path.get(out_path, set())
            for item in items:
                key = (alias, repeat_idx, item.question_id, str(args.prompt))
                if key in completed and not args.rerun_selected:
                    reporter.tick(skipped=True, detail=f"{alias} repeat={repeat_idx} {item.question_id}")
                    continue
                user_text = build_user_text(item)
                image_paths = resolve_image_paths(bench_root, item)
                row = {
                    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model_alias": alias,
                    "provider": spec["provider"],
                    "model": spec["model"],
                    "repeat_idx": repeat_idx,
                    "question_id": item.question_id,
                    "question_type": item.question_type,
                    "domain": item.domain,
                    "image_based": item.image_based,
                    "image_paths": item.image_paths,
                    "prompt_file": str(args.prompt),
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "response_text": "",
                    "parsed_answer": "",
                    "parsed_explanation": "",
                    "parsed_response_valid": 0,
                    "error": "",
                    "response_metadata": {},
                }
                if args.dry_run:
                    row["response_text"] = "[DRY RUN]"
                else:
                    for attempt in range(args.retries + 1):
                        try:
                            response = provider.generate(
                                system_prompt=system_prompt,
                                user_text=user_text,
                                image_paths=image_paths,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                            )
                            row["response_text"] = response.text
                            row["response_metadata"] = summarize_raw_response(response.raw)
                            parsed_response = parse_model_response(response.text, item.question_type)
                            row["parsed_answer"] = parsed_response["answer"]
                            row["parsed_explanation"] = parsed_response["explanation"]
                            row["parsed_response_valid"] = 1 if parsed_response["answer"] and parsed_response.get("json_valid") == "1" else 0
                            row["error"] = ""
                            if row["response_metadata"].get("finish_reason") == "length":
                                row["error"] = "RuntimeError('Provider response was truncated: finish_reason=length')"
                                if attempt < args.retries:
                                    time.sleep(args.retry_sleep)
                                    continue
                            elif not response.text.strip():
                                row["error"] = "RuntimeError('Empty response_text from provider')"
                                if attempt < args.retries and is_retryable_empty_response(row["response_metadata"]):
                                    time.sleep(args.retry_sleep)
                                    continue
                            elif not row["parsed_response_valid"] and attempt < args.retries:
                                row["error"] = "RuntimeError('Could not parse valid JSON answer from provider response')"
                                time.sleep(args.retry_sleep)
                                continue
                            break
                        except Exception as exc:  # noqa: BLE001 - batch runs should continue.
                            row["error"] = repr(exc)
                            if attempt >= args.retries or not is_retryable_error(exc):
                                break
                            time.sleep(args.retry_sleep)
                    row["attempts"] = attempt + 1
                append_jsonl(out_path, row)
                reporter.tick(
                    error=bool(row["error"]),
                    detail=f"{alias} repeat={repeat_idx} {item.question_id}",
                )
                if args.sleep:
                    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
