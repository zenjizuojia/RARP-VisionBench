# RARP-VisionBench

RARP-VisionBench is a 100-question benchmark for evaluating vision-language models (VLMs) in robot-assisted radical prostatectomy (RARP). It is designed for reproducible assessment of surgical scene understanding, procedural reasoning, operative safety knowledge, and RARP-related clinical decision-making.

This repository contains the benchmark data, image assets, baseline prompts, model-running scripts, LLM-based judging scripts, and utilities for generating summary matrices. It does not include local model outputs or judged result files.

## Study Overview

Vision-language models may support surgical education and operative decision support, but their ability to interpret RARP scenes remains uncertain. RARP-VisionBench provides a RARP-specific assessment framework covering anatomical landmark recognition, surgical step identification and sequencing, operative safety and technical principles, and clinical decision-making for RARP eligibility.

The benchmark supports both answer-only evaluation and answer-with-explanation evaluation. The primary endpoint is item-level accuracy. The secondary endpoint is explanation reliability, scored by an LLM judge on a 0-2 scale.

## Repository Structure

```text
RARP-VisionBench/
├── data/
│   ├── questions_full.jsonl
│   ├── questions_full.csv
│   ├── questions_full.xlsx
│   ├── answer_key.json
│   ├── questions_smoke.jsonl
│   ├── questions_20t.jsonl
│   ├── answer_key_20t.json
│   ├── questions_full_external.jsonl
│   ├── questions_full_external.csv
│   ├── questions_full_external.xlsx
│   ├── answer_key_external.json
│   ├── image_metadata_external.csv
│   └── image_sources_external.csv
├── images/
│   ├── annotated/
│   ├── raw_extracted/
│   └── external/
├── prompts/
│   ├── direct_answer.txt
│   └── structured_rationale.txt
├── docs/
│   ├── benchmark_card.md
│   └── scoring_rubric.md
├── eval/
│   ├── run_eval.py
│   ├── judge_outputs.py
│   ├── summarize_judged_outputs.py
│   ├── compact_model_outputs.py
│   └── rarpbench/
├── results/
│   ├── model_outputs/
│   ├── judged_outputs/
│   ├── model_outputs_20t/
│   └── judged_outputs_20t/
├── .env.example
├── requirements.txt
└── requirements-optional.txt
```

## Benchmark Files

The main benchmark is provided in three formats:

- `data/questions_full.jsonl`: machine-readable version for batch VLM evaluation
- `data/questions_full.csv`: tabular version for inspection
- `data/questions_full.xlsx`: editable spreadsheet version

The reference answers are stored separately in `data/answer_key.json`.

Additional subsets:

- `data/questions_20t.jsonl`: 20-question subset for repeated-temperature or repeated-sampling experiments
- `data/answer_key_20t.json`: answer key for the 20-question subset
- `data/questions_full_external.jsonl`: 30-item external validation set (see below)
- `data/answer_key_external.json`: answer key for the external validation set

## Question Domains

RARP-VisionBench contains 100 items across four RARP-relevant domains:

| Domain | Number of Items |
| --- | ---: |
| Anatomical landmark recognition | 30 |
| Surgical step identification and sequencing | 30 |
| Operative safety and technical principles | 30 |
| Clinical decision-making for RARP eligibility | 10 |

Question types include single-choice, multiple-choice, true/false, and free-text/open-ended questions. Multiple-choice items are scored at the whole-item level.

## Data Format

Each JSONL row in `questions_full.jsonl` represents one benchmark item. Common fields include:

- `question_id`: stable item identifier
- `domain`: benchmark domain
- `question_type`: item type
- `question_text`: question stem
- `options`: answer options when applicable
- `image_paths`: relative image paths, if the item uses one or more images
- `image_based`: whether image interpretation is required

Model outputs are expected to be JSON-formatted by default. The evaluation scripts parse:

```json
{
  "answer": "A",
  "explanation": "Brief explanation here."
}
```

For answer-only prompts, the `explanation` field may be empty.

## Installation

The runtime evaluation pipeline uses only the Python standard library.

```bash
git clone <your-repository-url>
cd RARP-VisionBench
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

No required packages need to be installed for normal benchmark runs. Optional document/spreadsheet rebuilding dependencies are listed in `requirements-optional.txt`.

## API Configuration

Copy `.env.example` to `.env` and fill in the keys you need:

```bash
cp .env.example .env
```

Example environment variables:

```text
DASHSCOPE_API_KEY=your_dashscope_api_key
OUYIAI_API_KEY=your_ouyiai_api_key
AIHUBMIX_API_KEY=your_aihubmix_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

Model definitions are configured in:

```text
eval/config/models.example.json
```

You can edit model aliases, model names, base URLs, and API-key environment variable names as needed. The provider layer supports OpenAI-compatible chat-completions APIs.

## Full Benchmark Evaluation

Run answer-only evaluation with four repeats:

```bash
python eval/run_eval.py \
  --questions data/questions_full.jsonl \
  --prompt prompts/direct_answer.txt \
  --models qwen3-vl-plus-aliyun gpt-5-aihubmix gemini-3-flash-preview-ouyiai \
  --repeats 4 \
  --temperature 0 \
  --max-tokens 2048 \
  --progress-every 10 \
  --retries 5 \
  --retry-sleep 8
```

Run one answer-with-explanation pass:

```bash
python eval/run_eval.py \
  --questions data/questions_full.jsonl \
  --prompt prompts/structured_rationale.txt \
  --models qwen3-vl-plus-aliyun gpt-5-aihubmix gemini-3-flash-preview-ouyiai \
  --repeats 1 \
  --temperature 0 \
  --max-tokens 2048 \
  --progress-every 10 \
  --retries 5 \
  --retry-sleep 8
```

The runner supports resume behavior. If an output file already contains completed rows, rerunning the same command skips completed items unless `--force` is provided.

## LLM-Based Judging

Answer-only outputs are judged for binary correctness:

```bash
python eval/judge_outputs.py \
  --mode answer \
  --outputs-glob "results/model_outputs/*direct_answer.jsonl" \
  --models qwen3-vl-plus-aliyun gpt-5-aihubmix gemini-3-flash-preview-ouyiai \
  --progress-every 10
```

Answer-with-explanation outputs are judged for both answer correctness and explanation reliability:

```bash
python eval/judge_outputs.py \
  --mode rationale \
  --outputs-glob "results/model_outputs/*structured_rationale.jsonl" \
  --models qwen3-vl-plus-aliyun gpt-5-aihubmix gemini-3-flash-preview-ouyiai \
  --progress-every 10
```

Judged outputs are written to `results/judged_outputs/`. The default judge is configured as an OpenAI-compatible DeepSeek endpoint. You can change `--base-url`, `--judge-model`, and `--api-key-env`.

## Summary Matrices

After judging, generate Excel matrices:

```bash
python eval/summarize_judged_outputs.py \
  --models qwen3-vl-plus-aliyun gpt-5-aihubmix gemini-3-flash-preview-ouyiai \
  --out results/judged_outputs/rarp_visionbench_summary_matrices.xlsx
```

The workbook includes per-model sheets for accuracy, explanation reliability, and answer-correctness versus explanation-reliability comparison.

For the 20-question repeated subset:

```bash
python eval/run_eval.py \
  --questions data/questions_20t.jsonl \
  --prompt prompts/direct_answer.txt \
  --out-dir results/model_outputs_20t \
  --models qwen3-vl-plus-aliyun gpt-5-aihubmix gemini-3-flash-preview-ouyiai \
  --repeats 5 \
  --temperature 0 \
  --max-tokens 2048 \
  --progress-every 5

python eval/judge_outputs.py \
  --questions data/questions_20t.jsonl \
  --answers data/answer_key_20t.json \
  --outputs-glob "results/model_outputs_20t/*.jsonl" \
  --out-dir results/judged_outputs_20t \
  --models qwen3-vl-plus-aliyun gpt-5-aihubmix gemini-3-flash-preview-ouyiai \
  --mode answer \
  --progress-every 5

python eval/summarize_judged_outputs.py \
  --questions data/questions_20t.jsonl \
  --answers data/answer_key_20t.json \
  --judged-dir results/judged_outputs_20t \
  --models qwen3-vl-plus-aliyun gpt-5-aihubmix gemini-3-flash-preview-ouyiai \
  --out results/judged_outputs_20t/rarp_20t_summary_matrices.xlsx
```

## Output Cleaning

If model-output files contain duplicate rows from interrupted or repeated runs, compact them before judging:

```bash
python eval/compact_model_outputs.py --glob "results/model_outputs/*.jsonl" --dry-run
python eval/compact_model_outputs.py --glob "results/model_outputs/*.jsonl"
```

The dry run reports how many rows would remain after compaction.

## Scoring

See `docs/scoring_rubric.md` for the scoring policy. In brief:

- Accuracy is scored as 1 for correct and 0 for incorrect.
- Multiple-choice items require the complete correct option set.
- Free-text answers are judged by semantic equivalence against the reference answer.
- Explanation reliability is scored as 0, 1, or 2.

The LLM judge receives the question text, model answer, model explanation when available, and reference answer. Benchmark images are not sent to the judge by default.

## External Validation Set

An independent external validation set built from the public RARPL dataset (Zenodo `10.5281/zenodo.15228400`, CC BY 4.0; 10 patients, 32 clips, da Vinci Xi) is provided in the same format as the development benchmark:

- `data/questions_full_external.jsonl` / `.csv` / `.xlsx`: 30 items (15 anatomical free-text and 15 procedural), with item IDs prefixed `E-`.
- `data/answer_key_external.json`: reference answers and scoring rules.
- `data/image_metadata_external.csv`: mapping of `images/external/image1`–`image47` to items.
- `data/image_sources_external.csv`: provenance of every external image (RARPL clip and frame index).
- `images/external/`: 47 annotated frames.

The external set is image-only by design: it covers the anatomical and procedural domains but contains no text-only items and no clinical decision-making items, because those require guideline material and patient-level clinical data that a public video dataset does not provide. It is always analysed separately and is never pooled with the 100 development items.

The same scripts are used, with the external questions and answer key:

```bash
python eval/run_eval.py --questions data/questions_full_external.jsonl \
  --prompt prompts/direct_answer.txt --repeats 5 \
  --out-dir results/model_outputs_external

python eval/judge_outputs.py --questions data/questions_full_external.jsonl \
  --answers data/answer_key_external.json \
  --outputs-glob "results/model_outputs_external/*.jsonl" --mode answer \
  --out-dir results/judged_outputs_external
```

## Ethical and Clinical Use

RARP-VisionBench is intended for research and educational evaluation of VLMs. It is not a medical device and should not be used for unsupervised clinical decision-making. Model outputs should be interpreted by qualified clinicians, especially when surgical safety or patient-specific decisions are involved.

## Citation

If you use RARP-VisionBench, please cite the associated manuscript once available:

```text
RARP-VisionBench: evaluating vision-language models in robot-assisted radical prostatectomy.
```

## License

No license is included by default. Add an appropriate license before public release.
