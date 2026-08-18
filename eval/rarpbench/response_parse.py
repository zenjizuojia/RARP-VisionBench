from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def normalize_answer(answer: str, question_type: str) -> str:
    answer = str(answer).strip()
    if question_type in {"single_choice", "multiple_choice", "sequencing"}:
        letters = re.findall(r"[A-Z]", answer.upper())
        if question_type == "multiple_choice":
            return "".join(sorted(dict.fromkeys(letters)))
        return "".join(letters)
    if question_type == "true_false":
        normalized = answer.strip().upper()
        if normalized in {"TRUE", "T", "YES"}:
            return "T"
        if normalized in {"FALSE", "F", "NO"}:
            return "F"
    return answer


def parse_model_response(text: str, question_type: str = "") -> dict[str, str]:
    parsed = extract_json_object(text)
    if parsed:
        answer = str(parsed.get("answer", "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()
        json_valid = "1"
    else:
        answer = text.strip()
        explanation = ""
        json_valid = "0"
    return {
        "answer": normalize_answer(answer, question_type),
        "explanation": explanation,
        "json_valid": json_valid,
    }
