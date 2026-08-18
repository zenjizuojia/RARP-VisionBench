# Evaluation Code

This folder contains the evaluation implementation.

For normal use, follow the Chinese guide at:

```text
../中文运行说明.md
```

Main scripts:

- `run_eval.py`: runs VLMs on the benchmark and writes split JSONL outputs to `results/model_outputs/`.
- `judge_outputs.py`: uses DeepSeek or another OpenAI-compatible text LLM to judge model outputs and writes JSONL/CSV files to `results/judged_outputs/`.
- `score_outputs.py`: legacy rule-based scorer; DeepSeek judging is recommended for the current workflow.

Core modules:

- `rarpbench/data.py`: benchmark loading.
- `rarpbench/prompts.py`: prompt construction.
- `rarpbench/providers/`: OpenAI-compatible provider adapters.
- `rarpbench/progress.py`: progress display and resume helpers.

