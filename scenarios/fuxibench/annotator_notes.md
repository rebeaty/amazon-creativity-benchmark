# Annotator Requirements: FuxiBench (poem_gen, poem_nmt_inv)

Source: Paper Section 4.2, Figure 4; `evaluate.py` (`lacc_evaluation`)

## Configuration for LLM-as-Judge

Judge model: Qwen2-7B-Instruct (fine-tuned on 2,000 human-annotated examples)
Output: Binary Y/N with reasoning
Aggregation: Percentage of Y judgments (lacc score)

## Judge Criteria (from Figure 4)

The judge evaluates whether the model output is correct based on:
1. 包含要点 — Contains key points from the reference answer
2. 与标准答案一致 — Consistency with the standard answer
3. 符合问题和任务要求 — Adherence to task requirements
4. 符合事实 — Factual accuracy
5. 预测答案简洁明了 — Concise and clear

## Validation

- Tested on 1,000 human-annotated samples
- Accuracy: 89.8%
- Cohen's kappa: 0.764 (substantial agreement)

## Alternative API Path

The code also supports `lacc_evaluation_api()` via LangChain with `CriteriaResultOutputParserZH()` using `general_criteria`, allowing any LLM as judge. This is the fallback when the fine-tuned Qwen2 model is not available.

## Notes

- All evaluation is in Chinese
- For poem_gen: judge checks if generated poetry matches the requested keywords/theme
- For poem_nmt_inv: judge checks if reconstructed classical poem matches the reference original
