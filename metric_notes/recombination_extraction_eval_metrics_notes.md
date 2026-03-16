# Metric Requirements: Recombination Extraction

Source: CHIMERA paper (arXiv:2505.20779), Section 4 and Appendix; PROMPT_E2E from src/util.py

## Overview

The Recombination Extraction benchmark evaluates models at three levels using a **single unified prompt** (PROMPT_E2E):
1. **Abstract Classification**: Binary classification (relevant vs irrelevant) via JSON output
2. **Entity Extraction**: Extracting recombination elements from text
3. **Relation Extraction**: Identifying recombination type and linking entities

**Key insight**: All three levels use the same prompt asking for structured JSON extraction. Evaluation differs by what aspect of the output is checked.

## Current HELM Implementation

The scenario file (`recombination_extraction_scenario.py`) implements **Level 1 only**: binary classification.

### Level 1: Binary Classification via JSON Parsing

**Prompt**: Uses PROMPT_E2E that asks models to output recombination as JSON:
- If recombination present: `{"combination": {"comb-element": [...]}}`  OR `{"inspiration": {"inspiration-src": [...], "inspiration-target": [...]}}`
- If no recombination: `{}`

**Evaluation Logic**:
- Empty JSON `{}` → classified as "irrelevant"
- Non-empty JSON → classified as "relevant"

**HELM Evaluation**:
- Currently uses References with `{}` for irrelevant and example JSON for relevant
- **Note**: Requires custom metric or post-processing to properly parse JSON and determine if empty
- Simple `exact_match` won't work due to JSON format variations (whitespace, ordering, etc.)
- **Recommendation**: Implement custom metric that:
  1. Parses model JSON output (handling malformed JSON)
  2. Checks if parsed JSON is empty dict or has content
  3. Compares against ground truth document_class

## Full Benchmark Evaluation (Future Implementation)

### Level 2: Entity Extraction

**Task**: Extract entities involved in recombination from abstracts marked as "relevant"

**Entity Types**:
- `analogy-src`: Source domain for inspiration/analogy
- `analogy-target`: Target domain receiving inspiration
- `comb-element`: Elements being combined in a blend

**Example**:
```json
{
  "text": "Many works in robot teaching... However, to effectively teach a complex task sequence to a robot, it is important to take advantage of both task and motion knowledge...",
  "entities": {
    "analogy-src": ["facts on human body motion"],
    "analogy-target": ["teach a complex task sequence to a robot"]
  }
}
```

**Evaluation Methodology**:
- Uses **soft matching** with semantic similarity
- GPT-4o-mini as judge to determine if two entities refer to the same concept
- Metrics: Precision, Recall, F1

**Implementation Requirements**:
- Custom metric that:
  1. Parses structured JSON output from model
  2. Calls GPT-4o-mini to compare extracted vs. gold entities
  3. Computes F1 with soft matching

### Level 3: Relation Extraction

**Task**: Identify the type of recombination and link entities

**Relation Types**:
- `analogy` (inspiration): Links `analogy-src` → `analogy-target`
- `combination` (blend): Links multiple `comb-element` entities

**Example**:
```json
{
  "relations": {
    "analogy": [{
      "analogy-src": ["facts on human body motion"],
      "analogy-target": ["teach a complex task sequence to a robot"]
    }]
  }
}
```

**Evaluation Methodology**:
- Same soft matching approach as entities
- Relation matches if: (1) relation type is correct AND (2) all linked entities match
- Metrics: Precision, Recall, F1

## Performance Benchmarks (from paper)

### Human Performance (Inter-annotator Agreement):
- Classification: F1 = 0.912
- Entity Extraction: Lower (entities and relations are more challenging)
- Relation Extraction: Lower

### Model Performance (Mistral-7B fine-tuned):
- Best automatic system across all subtasks
- Still significantly below human performance on entity/relation extraction

## Notes for Future Implementation

1. **Structured Output**: Models must generate JSON with entities and relations
2. **LLM-as-Judge**: Requires GPT-4o-mini API access for soft matching
3. **Multi-task Evaluation**: Could be split into separate scenarios for each level
4. **Soft Matching Definition**: Two entities match if semantically similar (paper Appendix B)

## References

- Paper: https://arxiv.org/abs/2505.20779
- Code: https://github.com/noy-sternlicht/CHIMERA-KB
- Data: https://huggingface.co/datasets/noystl/Recombination-Extraction
- Annotation Guidelines: See `annotation_guidelines.pdf` in GitHub repo
