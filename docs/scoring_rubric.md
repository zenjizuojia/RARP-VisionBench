# Scoring Rubric

## Primary Endpoint: Overall Accuracy

For the primary benchmark endpoint, each of the 100 items is converted to binary correctness:

`overall_accuracy = number_of_correct_items / 100`

This endpoint is intended for model-to-model comparison and should be reported for the full benchmark and for predefined subgroups.

## Item-Level Rules

- Single-choice: correct if the selected option letter exactly matches the reference answer.
- True/false: correct if the response exactly matches the reference true/false label.
- Sequencing: correct if the full sequence exactly matches the reference sequence.
- Multiple-choice: correct if the selected option set exactly matches the reference option set.
- Free-text anatomical identification: correct if the response identifies the same clinically relevant structure in the question context, including accepted synonyms, abbreviations, and contextually equivalent broader/narrower operative terms.
- Open-ended operative-safety questions: correct if the response covers all essential key points in clinically equivalent language and contains no major clinical error.

## Clinical Equivalence for Free-Text/Open-Ended Items

Free-text and open-ended answers should not be graded by exact string matching. The grader should accept clinically equivalent expressions when they refer to the same target structure, step, or principle in the RARP context.

Examples of acceptable contextual equivalence include:

- Urethra, membranous urethra, posterior urethra, urethral stump, and urethral sphincter region when the item targets the urethral/apical anastomotic region.
- Bladder neck and vesicourethral junction when the item targets the anastomotic site.
- DVC, dorsal venous complex, and Santorini plexus.
- Space of Retzius and retropubic space.
- Urachus and median umbilical fold/ligament.
- Medial umbilical fold and obliterated umbilical artery.
- Seminal vesicle and SV.

Do not accept responses that name a different structure, are too vague for the image/question, omit an essential required element, or introduce a major clinical contradiction.

## Secondary Endpoint: Explanation Quality

When models are prompted to provide a rationale, score explanation quality separately from answer correctness.

Recommended 0-2 scale:

- 0: Incorrect, irrelevant, misleading, or absent rationale.
- 1: Partially correct rationale with incomplete support or minor errors.
- 2: Clinically coherent rationale that supports the final answer without major error.

Recommended reporting:

- Mean explanation-quality score.
- Proportion of correct answers with flawed rationale.
- Proportion of incorrect answers with plausible but incorrect rationale.
- Critical clinical error rate in explanations.

## Adjudication

Ambiguous free-text and open-ended responses should be adjudicated by at least two independent urology/RARP experts. Disagreements should be resolved by consensus or a third expert. Inter-rater reliability should be reported for the final benchmark study.
