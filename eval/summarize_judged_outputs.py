from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_LABELS = {
    "qwen3-vl-flash-aliyun": "qwen",
    "llama-4-maverick-aihubmix": "llama",
    "gpt-4o-mini-aihubmix": "gpt4o",
    "qwen3-vl-plus-aliyun": "qwenplus",
    "gpt-5-aihubmix": "gpt5",
    "gemini-3-flash-preview-ouyiai": "gemini3",
}


META_COLUMNS = [
    "question_id",
    "domain",
    "question_type",
    "image_based",
    "question_text",
    "options_text",
    "reference_answer",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    questions = {}
    for row in read_jsonl(path):
        questions[row["question_id"]] = row
    return questions


def load_answer_key(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def judged_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model_alias"),
        row.get("repeat_idx"),
        row.get("question_id"),
        row.get("judge_mode"),
        row.get("judge_model"),
    )


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = judged_key(row)
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]


def load_judged_outputs(root: Path, judged_dir: Path) -> tuple[dict[str, dict[int, dict[str, dict[str, Any]]]], dict[str, dict[str, dict[str, Any]]]]:
    answer: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    rationale: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted((root / judged_dir).glob("*.jsonl")):
        rows = latest_rows(read_jsonl(path))
        for row in rows:
            model = row.get("model_alias", "")
            qid = row.get("question_id", "")
            mode = row.get("judge_mode", "")
            repeat_idx = int(row.get("repeat_idx", 0) or 0)
            if mode == "answer":
                answer[model][repeat_idx][qid] = row
            elif mode == "rationale":
                rationale[model][qid] = row
    return answer, rationale


def question_meta(question: dict[str, Any], answer_key: dict[str, dict[str, Any]]) -> list[Any]:
    qid = question["question_id"]
    return [
        qid,
        question.get("domain", ""),
        question.get("question_type", ""),
        question.get("image_based", ""),
        question.get("question_text", ""),
        question.get("options_text", ""),
        answer_key.get(qid, {}).get("reference_answer", ""),
    ]


def as_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return 1 if value else 0
    return value


def build_accuracy_sheet(
    *,
    model: str,
    questions: dict[str, dict[str, Any]],
    answer_key: dict[str, dict[str, Any]],
    answer_rows: dict[str, dict[int, dict[str, dict[str, Any]]]],
    rationale_rows: dict[str, dict[str, dict[str, Any]]],
    direct_repeats: list[int],
) -> list[list[Any]]:
    headers = META_COLUMNS + [f"direct_repeat{repeat_idx:02d}_correct" for repeat_idx in direct_repeats]
    if rationale_rows.get(model):
        headers.append("rationale_repeat01_correct")
    rows = [headers]
    for qid, question in questions.items():
        model_answer = answer_rows.get(model, {})
        rationale = rationale_rows.get(model, {})
        row = question_meta(question, answer_key)
        row.extend([(model_answer.get(repeat_idx, {}).get(qid, {})).get("correct", "") for repeat_idx in direct_repeats])
        if rationale_rows.get(model):
            row.append((rationale.get(qid, {})).get("correct", ""))
        rows.append(row)
    return rows


def build_reliability_sheet(
    *,
    model: str,
    questions: dict[str, dict[str, Any]],
    answer_key: dict[str, dict[str, Any]],
    rationale_rows: dict[str, dict[str, dict[str, Any]]],
) -> list[list[Any]]:
    headers = META_COLUMNS + [
        "reliability_score",
        "model_answer",
        "model_explanation",
        "judge_reason",
        "judge_error",
    ]
    rows = [headers]
    rationale = rationale_rows.get(model, {})
    for qid, question in questions.items():
        judged = rationale.get(qid, {})
        row = question_meta(question, answer_key)
        row.extend([
            judged.get("reliability_score", ""),
            judged.get("model_answer", ""),
            judged.get("model_explanation", ""),
            judged.get("judge_reason", ""),
            judged.get("judge_error", ""),
        ])
        rows.append(row)
    return rows


def build_answer_vs_rationale_sheet(
    *,
    model: str,
    questions: dict[str, dict[str, Any]],
    answer_key: dict[str, dict[str, Any]],
    rationale_rows: dict[str, dict[str, dict[str, Any]]],
) -> list[list[Any]]:
    headers = META_COLUMNS + [
        "rationale_answer_correct",
        "reliability_score",
        "correct_answer_but_unreliable_explanation",
        "model_answer",
        "model_explanation",
        "judge_reason",
        "judge_error",
    ]
    rows = [headers]
    rationale = rationale_rows.get(model, {})
    for qid, question in questions.items():
        judged = rationale.get(qid, {})
        correct = judged.get("correct", "")
        reliability = judged.get("reliability_score", "")
        flag = 1 if correct == 1 and reliability in (0, 1) else 0 if correct != "" and reliability != "" else ""
        row = question_meta(question, answer_key)
        row.extend([
            correct,
            reliability,
            flag,
            judged.get("model_answer", ""),
            judged.get("model_explanation", ""),
            judged.get("judge_reason", ""),
            judged.get("judge_error", ""),
        ])
        rows.append(row)
    return rows


def column_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xml_text(value: Any) -> str:
    return html.escape(str(value), quote=False)


def sheet_xml(rows: list[list[Any]]) -> str:
    xml_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, raw_value in enumerate(row, start=1):
            value = as_cell(raw_value)
            ref = f"{column_name(c_idx - 1)}{r_idx}"
            if value == "":
                cells.append(f'<c r="{ref}"/>')
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{xml_text(value)}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '<autoFilter ref="A1:Z101"/>'
        '</worksheet>'
    )


def safe_sheet_name(name: str, used: set[str]) -> str:
    name = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31]
    base = name
    suffix = 1
    while name in used:
        tail = f"_{suffix}"
        name = f"{base[:31-len(tail)]}{tail}"
        suffix += 1
    used.add(name)
    return name


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    sheet_names = [safe_sheet_name(name, used) for name, _ in sheets]
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = "".join(
        f'<sheet name="{html.escape(name, quote=True)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{workbook_sheets}</sheets></workbook>"
    )
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx in range(1, len(sheets) + 1):
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheets)+1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr("xl/styles.xml", styles)
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize judged RARP-VisionBench outputs into matrix workbooks.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("."))
    parser.add_argument("--questions", type=Path, default=Path("data/questions_full.jsonl"))
    parser.add_argument("--answers", type=Path, default=Path("data/answer_key.json"))
    parser.add_argument("--judged-dir", type=Path, default=Path("results/judged_outputs"))
    parser.add_argument("--out", type=Path, default=Path("results/judged_outputs/rarp_visionbench_summary_matrices.xlsx"))
    parser.add_argument("--models", nargs="*", help="Optional model aliases to include in the workbook.")
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    questions = load_questions(root / args.questions)
    answer_key = load_answer_key(root / args.answers)
    answer_rows, rationale_rows = load_judged_outputs(root, args.judged_dir)

    known_models = [model for model in MODEL_LABELS if model in answer_rows or model in rationale_rows]
    extra_models = sorted((set(answer_rows) | set(rationale_rows)) - set(MODEL_LABELS))
    models = known_models + extra_models
    if args.models:
        selected = set(args.models)
        models = [model for model in models if model in selected]
    if not models:
        raise SystemExit("No judged output rows found.")

    sheets: list[tuple[str, list[list[Any]]]] = []
    for model in models:
        label = MODEL_LABELS.get(model, model[:12])
        direct_repeats = sorted(answer_rows.get(model, {}))
        sheets.append((f"accuracy_{label}", build_accuracy_sheet(
            model=model,
            questions=questions,
            answer_key=answer_key,
            answer_rows=answer_rows,
            rationale_rows=rationale_rows,
            direct_repeats=direct_repeats,
        )))
    for model in models:
        if not rationale_rows.get(model):
            continue
        label = MODEL_LABELS.get(model, model[:12])
        sheets.append((f"reliability_{label}", build_reliability_sheet(
            model=model,
            questions=questions,
            answer_key=answer_key,
            rationale_rows=rationale_rows,
        )))
    for model in models:
        if not rationale_rows.get(model):
            continue
        label = MODEL_LABELS.get(model, model[:12])
        sheets.append((f"answer_vs_expl_{label}", build_answer_vs_rationale_sheet(
            model=model,
            questions=questions,
            answer_key=answer_key,
            rationale_rows=rationale_rows,
        )))

    output_path = root / args.out
    write_xlsx(output_path, sheets)
    for name, rows in sheets:
        print(f"{name}: rows={len(rows)-1} cols={len(rows[0])}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
