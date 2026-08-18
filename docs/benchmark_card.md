# Benchmark Card: RARP-VisionBench v0.1

## Intended Use

Evaluate vision-language model performance on static RARP scenario understanding, including anatomical landmark recognition, surgical step understanding, operative safety principles, and clinical decision-making.

## Data Source

The current release is derived from the original RARP Simulation Test supplementary Word document. Images are annotated screenshots embedded in the source document.

## Important Scope Limitation

This benchmark uses static annotated images. It does not evaluate real-time video understanding, instrument tracking, temporal reasoning, or safe intraoperative deployment.

## Recommended Reporting

- Overall accuracy across 100 items.
- Accuracy by domain.
- Accuracy by question type.
- Accuracy for image-based vs text-only questions.
- Explanation-quality score and critical-error rate when rationales are collected.
