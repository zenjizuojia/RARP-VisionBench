from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rarpbench.progress import ProgressReporter, count_jsonl_rows, iter_jsonl, load_completed_keys
from rarpbench.response_parse import extract_json_object, parse_model_response


ANSWER_ONLY_SYSTEM = """You are a strict but fair medical examination grader for robot-assisted radical prostatectomy (RARP).

You will receive a question, a model answer, and the reference answer.
Judge whether the model answer is clinically equivalent to the reference answer.

Return exactly one valid JSON object with exactly these keys:
{
  "correct": 0 or 1,
  "reason": "<brief reason, 20 words or fewer>"
}

Rules:
- Use compact JSON. Use integer 1 for correct and integer 0 for incorrect. Do not use true/false, yes/no, strings, markdown, or comments outside JSON.
- The reason must be short: 20 words or fewer.
- For single-choice, multiple-choice, true/false, and sequencing questions, compare the final answer strictly.
- For multiple-choice, ignore order but require the same option set.
- For free-text anatomical identification and open-ended operative questions, grade by clinical equivalence, not exact wording.
- Give 1 if the model answer identifies the same clinically relevant structure, step, or principle using an accepted synonym, abbreviation, broader/narrower term that is equivalent in the question context, or a common operative description.
- Give 0 if the answer names a different structure, is too vague for the question, omits an essential required element, contradicts the reference, or contains a major clinical error.
- RARP equivalence examples that may be correct when context supports them: urethra / membranous urethra / posterior urethra / urethral stump / urethral sphincter region; bladder neck / vesicourethral junction when the question targets the anastomotic site; DVC / dorsal venous complex / Santorini plexus; Space of Retzius / retropubic space; urachus / median umbilical fold; medial umbilical fold / obliterated umbilical artery; seminal vesicle / SV.
- Do not require the exact reference wording if the model answer is clinically acceptable in the visual or operative context.
- Do not solve the question independently; grade against the reference answer.
"""


RATIONALE_SYSTEM = """You are a strict but fair medical examination grader for robot-assisted radical prostatectomy (RARP).

You will receive a question, a model answer, a model explanation, and the reference answer.
Judge the answer correctness and the reliability of the explanation.

Return exactly one valid JSON object with exactly these keys:
{
  "correct": 0 or 1,
  "reliability_score": 0, 1, or 2,
  "reason": "<brief reason, 20 words or fewer>"
}

Scoring:
- correct: 1 if the final answer is clinically equivalent to the reference answer, otherwise 0.
- reliability_score 0: completely unreliable, irrelevant, misleading, or contains a major clinical error.
- reliability_score 1: partially reliable; some correct reasoning but incomplete, weak, or with minor errors.
- reliability_score 2: fully reliable; clinically coherent and supports the answer without major error.

Important:
- Use compact JSON. Use integers only. Do not use true/false, yes/no, strings, markdown, or comments outside JSON.
- The reason must be short: 20 words or fewer.
- For single-choice, multiple-choice, true/false, and sequencing answers, compare the final answer strictly; for multiple-choice, ignore order but require the same option set.
- For free-text anatomical identification and open-ended operative questions, grade answer correctness by clinical equivalence, not exact wording.
- Accept clinically equivalent terms in RARP context, including common abbreviations and contextually equivalent broader/narrower terms.
- RARP equivalence examples that may be correct when context supports them: urethra / membranous urethra / posterior urethra / urethral stump / urethral sphincter region; bladder neck / vesicourethral junction when the question targets the anastomotic site; DVC / dorsal venous complex / Santorini plexus; Space of Retzius / retropubic space; urachus / median umbilical fold; medial umbilical fold / obliterated umbilical artery; seminal vesicle / SV.
- Grade against the reference answer and the provided question.
- Do not require extra details beyond what the question asks.
- Do not solve the question independently; evaluate whether the response is acceptable.
"""


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


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


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    raw = post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
    )
    return raw["choices"][0]["message"].get("content", "")


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    questions = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                questions[row["question_id"]] = row
    return questions


def load_answer_key(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_vlm_response(text: str, question_type: str = "") -> dict[str, str]:
    return parse_model_response(text, question_type)


def build_answer_only_prompt(
    question: dict[str, Any],
    model_answer: str,
    reference_meta: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "question": question["question_text"],
            "options": question.get("options_text", ""),
            "model_answer": model_answer,
            "reference_answer": reference_meta.get("reference_answer", ""),
            "accepted_answers": reference_meta.get("accepted_answers", ""),
            "scoring_rule": reference_meta.get("scoring_rule", ""),
            "grader_instruction": "Return exactly JSON with integer correct and brief reason. Apply strict option matching for choice/sequencing questions and clinical equivalence for free-text/open-ended questions.",
        },
        ensure_ascii=False,
        indent=2,
    )


def build_rationale_prompt(
    question: dict[str, Any],
    model_answer: str,
    model_explanation: str,
    reference_meta: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "question": question["question_text"],
            "options": question.get("options_text", ""),
            "model_answer": model_answer,
            "model_explanation": model_explanation,
            "reference_answer": reference_meta.get("reference_answer", ""),
            "accepted_answers": reference_meta.get("accepted_answers", ""),
            "scoring_rule": reference_meta.get("scoring_rule", ""),
            "grader_instruction": "Return exactly JSON with integer correct, integer reliability_score, and brief reason. Apply strict option matching for choice/sequencing questions and clinical equivalence for free-text/open-ended questions.",
        },
        ensure_ascii=False,
        indent=2,
    )


def as_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "correct", "right"}:
            return 1
        if normalized in {"false", "no", "incorrect", "wrong"}:
            return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_judge_json(judge_json: dict[str, Any], mode: str) -> dict[str, Any]:
    normalized = dict(judge_json)
    if "correct" not in normalized:
        for key in ("is_correct", "answer_correct", "score", "accuracy"):
            if key in normalized:
                normalized["correct"] = normalized[key]
                break
    if mode == "rationale" and "reliability_score" not in normalized:
        for key in ("reliability", "explanation_score", "rationale_score", "explanation_reliability"):
            if key in normalized:
                normalized["reliability_score"] = normalized[key]
                break
    if "reason" not in normalized:
        for key in ("rationale", "explanation", "judge_reason", "comment"):
            if key in normalized:
                normalized["reason"] = normalized[key]
                break
    return normalized


def validate_judge_json(judge_json: dict[str, Any], mode: str, raw_text: str) -> tuple[dict[str, Any], str]:
    judge_json = normalize_judge_json(judge_json, mode)
    correct = as_int(judge_json.get("correct"))
    if correct is None:
        return judge_json, f"Judge response missing valid integer correct field: {raw_text[:200]}"
    if mode == "rationale":
        reliability = as_int(judge_json.get("reliability_score"))
        if reliability is None:
            return judge_json, f"Judge response missing valid integer reliability_score field: {raw_text[:200]}"
    return judge_json, ""


def parse_and_validate_judge_response(raw_text: str, mode: str) -> dict[str, Any]:
    if not raw_text.strip():
        raise RuntimeError("Empty judge response")
    judge_json = extract_json_object(raw_text)
    if not judge_json:
        raise RuntimeError(f"Judge response did not contain valid JSON: {raw_text[:200]}")
    judge_json, validation_error = validate_judge_json(judge_json, mode, raw_text)
    if validation_error:
        raise RuntimeError(validation_error)
    return judge_json


def looks_like_secret(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("sk-", "sk_", "ak-", "pk-")) or len(value) > 40


def get_api_key(value: str) -> str:
    if looks_like_secret(value):
        return value
    return os.environ.get(value, "")


def write_csv(path: Path, rows: list[dict[str, Any]], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_alias",
        "repeat_idx",
        "question_id",
        "question_type",
        "domain",
        "image_based",
        "model_answer",
        "reference_answer",
        "correct",
    ]
    if mode == "rationale":
        fields.extend(["model_explanation", "reliability_score"])
    fields.extend(["judge_reason", "judge_error"])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def judge_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("judge_mode"),
        row.get("judge_model"),
        row.get("model_alias"),
        row.get("repeat_idx"),
        row.get("question_id"),
    )


def load_completed_judge_keys(path: Path, mode: str, *, include_errors: bool = False) -> set[tuple[Any, ...]]:
    completed: set[tuple[Any, ...]] = set()
    if not path.exists():
        return completed
    for row in iter_jsonl(path):
        if row.get("judge_error") and not include_errors:
            continue
        if row.get("correct") is None:
            continue
        if mode == "rationale" and row.get("reliability_score") is None:
            continue
        completed.add(judge_key(row))
    return completed


def output_stem_for_input(input_path: Path, mode: str) -> str:
    return f"{input_path.stem}_judged_{mode}"


def collect_input_paths(
    bench_root: Path,
    outputs: Path | None,
    outputs_glob: str,
    model_aliases: list[str] | None = None,
) -> list[Path]:
    if outputs is not None:
        return [bench_root / outputs]
    paths = sorted((bench_root).glob(outputs_glob))
    if model_aliases:
        wanted = tuple(f"{alias}_" for alias in model_aliases)
        paths = [path for path in paths if path.name.startswith(wanted)]
    return paths


def model_output_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model_alias"),
        row.get("repeat_idx"),
        row.get("question_id"),
        row.get("prompt_file"),
    )


def load_latest_model_outputs(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if limit is not None and idx > limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            key = model_output_key(row)
            if key not in latest:
                order.append(key)
            latest[key] = row
    return [latest[key] for key in order]


def latest_judged_rows(path: Path, mode: str) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    if not path.exists():
        return []
    for row in iter_jsonl(path):
        if row.get("judge_mode") != mode:
            continue
        key = judge_key(row)
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]


def judge_one_file(
    *,
    bench_root: Path,
    input_path: Path,
    judged_jsonl: Path,
    judged_csv: Path,
    questions: dict[str, dict[str, Any]],
    answer_key: dict[str, dict[str, Any]],
    api_key: str,
    args: argparse.Namespace,
    system_prompt: str,
) -> None:
    judged_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        judged_jsonl.write_text("", encoding="utf-8")
    completed = set()
    if not args.force:
        completed = load_completed_judge_keys(judged_jsonl, args.mode, include_errors=args.skip_errors)
    input_rows = load_latest_model_outputs(input_path, limit=args.limit)
    input_total = len(input_rows)
    reporter = ProgressReporter(total=input_total, label=f"judge:{args.mode}", every=args.progress_every)
    print(
        f"[judge:{args.mode}] input={input_path} rows={input_total} "
        f"resume_skippable={len(completed)} judge_model={args.judge_model} out={judged_jsonl}",
        flush=True,
    )

    with judged_jsonl.open("a", encoding="utf-8") as out:
        for raw_row in input_rows:
            qid = raw_row["question_id"]
            existing_key = (
                args.mode,
                args.judge_model,
                raw_row.get("model_alias"),
                raw_row.get("repeat_idx"),
                qid,
            )
            if existing_key in completed:
                reporter.tick(skipped=True, detail=f"{raw_row.get('model_alias')} repeat={raw_row.get('repeat_idx')} {qid}")
                continue
            question = questions[qid]
            reference_meta = answer_key[qid]
            reference_answer = reference_meta["reference_answer"]
            parsed_response = {
                "answer": str(raw_row.get("parsed_answer", "")).strip(),
                "explanation": str(raw_row.get("parsed_explanation", "")).strip(),
            }
            if not parsed_response["answer"]:
                parsed_response = parse_vlm_response(raw_row.get("response_text", ""), raw_row.get("question_type", ""))
            model_answer = parsed_response["answer"]
            model_explanation = parsed_response["explanation"]

            if args.mode == "answer":
                user_prompt = build_answer_only_prompt(question, model_answer, reference_meta)
            else:
                user_prompt = build_rationale_prompt(question, model_answer, model_explanation, reference_meta)

            judge_error = ""
            judge_raw_text = ""
            judge_json: dict[str, Any] = {}
            vlm_error = str(raw_row.get("error", "")).strip()
            if vlm_error:
                judge_error = f"Skipped VLM API failure: {vlm_error}"
                judge_json = {"correct": None, "reason": "VLM output was not available because the model/API call failed."}
                if args.mode == "rationale":
                    judge_json["reliability_score"] = None
            elif args.dry_run:
                judge_json = {"correct": None, "reason": "[DRY RUN]"}
                if args.mode == "rationale":
                    judge_json["reliability_score"] = None
            else:
                for attempt in range(args.retries + 1):
                    try:
                        judge_raw_text = call_openai_compatible(
                            base_url=args.base_url,
                            api_key=api_key,
                            model=args.judge_model,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        judge_json = parse_and_validate_judge_response(judge_raw_text, args.mode)
                        judge_error = ""
                        break
                    except Exception as exc:  # noqa: BLE001 - batch judging should continue.
                        judge_error = repr(exc)
                        if attempt >= args.retries:
                            break
                        time.sleep(args.retry_sleep)

            correct = as_int(judge_json.get("correct"))
            reliability = as_int(judge_json.get("reliability_score")) if args.mode == "rationale" else None
            if reliability is not None:
                reliability = min(2, max(0, reliability))
            if correct is not None:
                correct = 1 if correct else 0
            judged = {
                **raw_row,
                "model_answer": model_answer,
                "model_explanation": model_explanation,
                "reference_answer": reference_answer,
                "correct": correct,
                "reliability_score": reliability,
                "judge_reason": str(judge_json.get("reason", "")),
                "judge_raw_text": judge_raw_text,
                "judge_error": judge_error,
                "judge_model": args.judge_model,
                "judge_mode": args.mode,
                "judge_attempts": attempt + 1 if not args.dry_run and not vlm_error else 0,
            }
            out.write(json.dumps(judged, ensure_ascii=False) + "\n")
            reporter.tick(
                error=bool(judge_error),
                detail=f"{raw_row.get('model_alias')} repeat={raw_row.get('repeat_idx')} {qid}",
            )
            if args.sleep:
                time.sleep(args.sleep)

    all_rows = latest_judged_rows(judged_jsonl, args.mode)
    write_csv(judged_csv, all_rows, args.mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Use DeepSeek or another OpenAI-compatible LLM to judge VLM outputs.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("."))
    parser.add_argument("--questions", type=Path, default=Path("data/questions_full.jsonl"))
    parser.add_argument("--answers", type=Path, default=Path("data/answer_key.json"))
    parser.add_argument("--outputs", type=Path, default=None, help="Optional single model-output JSONL file to judge.")
    parser.add_argument("--outputs-glob", default="results/model_outputs/*.jsonl", help="Glob for model-output files when --outputs is omitted.")
    parser.add_argument("--models", nargs="*", help="Optional model aliases to include when using --outputs-glob.")
    parser.add_argument("--mode", choices=["answer", "rationale"], required=True)
    parser.add_argument("--judged-jsonl", type=Path, default=None, help="Optional JSONL output path. Use only with --outputs.")
    parser.add_argument("--judged-csv", type=Path, default=None, help="Optional CSV output path. Use only with --outputs.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/judged_outputs"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3, help="Retry empty, invalid, or incomplete judge responses N times.")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Seconds to wait between judge retries.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore existing judged rows and rerun all selected rows.")
    parser.add_argument("--skip-errors", action="store_true", help="Treat existing judge_error rows as completed when resuming.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N input rows.")
    args = parser.parse_args()

    bench_root = args.benchmark_root.resolve()
    load_dotenv(bench_root / ".env")
    questions = load_questions(bench_root / args.questions)
    answer_key = load_answer_key(bench_root / args.answers)
    api_key = get_api_key(args.api_key_env)
    if not args.dry_run and not api_key:
        hint = args.api_key_env if not looks_like_secret(args.api_key_env) else "<redacted>"
        raise SystemExit(f"Missing judge API key. Set the key directly or via environment variable: {hint}")

    system_prompt = ANSWER_ONLY_SYSTEM if args.mode == "answer" else RATIONALE_SYSTEM
    input_paths = collect_input_paths(bench_root, args.outputs, args.outputs_glob, args.models)
    if not input_paths:
        raise SystemExit(f"No input files found. Checked --outputs-glob={args.outputs_glob}")
    if args.outputs is None and (args.judged_jsonl is not None or args.judged_csv is not None):
        raise SystemExit("--judged-jsonl/--judged-csv can only be used with --outputs.")

    for input_path in input_paths:
        if args.outputs is not None:
            stem = output_stem_for_input(input_path, args.mode)
            judged_jsonl = bench_root / (args.judged_jsonl or args.out_dir / f"{stem}.jsonl")
            judged_csv = bench_root / (args.judged_csv or args.out_dir / f"{stem}.csv")
        else:
            stem = output_stem_for_input(input_path, args.mode)
            judged_jsonl = bench_root / args.out_dir / f"{stem}.jsonl"
            judged_csv = bench_root / args.out_dir / f"{stem}.csv"
        judge_one_file(
            bench_root=bench_root,
            input_path=input_path,
            judged_jsonl=judged_jsonl,
            judged_csv=judged_csv,
            questions=questions,
            answer_key=answer_key,
            api_key=api_key,
            args=args,
            system_prompt=system_prompt,
        )


if __name__ == "__main__":
    main()
