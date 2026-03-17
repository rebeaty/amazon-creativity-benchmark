# Annotator Notes: Eval3DAIGC-198 3D Model Description

Source: https://arxiv.org/abs/2404.04363 (COLING 2025)
        https://github.com/yisuanwang/Idea23D

## Task

Given a model's description of a 3D object (generated from multi-view rendered
images), score the description against a human-annotated ground truth using an
LLM judge.

## Configuration for LLMAsJuryAnnotator

Judge model: GPT-4o (with vision)
Evaluation type: rubric-based scoring per instance
Approach: zero-shot rubric prompt

## Scoring Rubric (1–5 scale per criterion)

| # | Criterion | Description |
|---|-----------|-------------|
| 1 | Shape Accuracy | Does the description correctly identify the overall shape and structure of the 3D object? |
| 2 | Detail Completeness | Are important features (surface texture, colors, materials, distinctive elements) mentioned? |
| 3 | Spatial Understanding | Does the description demonstrate understanding of 3D spatial relationships (proportions, arrangement of parts, pose)? |
| 4 | Alignment with Ground Truth | How well does the description match the key content of the human-annotated reference description? |
| 5 | Descriptive Quality | Is the description clear, well-organized, and appropriately detailed (neither too sparse nor padded)? |

## Judge Prompt Template

```
You are evaluating a description of a 3D model. You are given:
1. The model-generated description
2. A human-written ground-truth description of the same 3D object

Score the model description on each criterion below (1-5, where 1=poor, 5=excellent):

1. Shape Accuracy: Correctly identifies overall shape and structure
2. Detail Completeness: Mentions important features (texture, color, materials)
3. Spatial Understanding: Demonstrates 3D spatial awareness (proportions, pose, arrangement)
4. Alignment with Ground Truth: Matches key content of the reference description
5. Descriptive Quality: Clear, well-organized, appropriately detailed

Ground Truth Description:
{reference}

Model Description:
{response}

Provide scores as: Shape: X, Detail: X, Spatial: X, Alignment: X, Quality: X
Then provide a brief justification.
```

## Notes

- The ground truth descriptions were manually annotated by the Idea23D paper
  authors. They describe the intended 3D model in detail, covering shape,
  appearance, pose, and notable features.
- Original paper metrics (CLIP similarity, ULIP-2) evaluate embedding-level
  alignment; the LLM judge provides a more interpretable quality assessment.
- Multi-object cases (31 of 132) have multiple view composites; the description
  should cover all objects in the scene.
- Open-ended metrics (BLEU, ROUGE, F1) provide automated reference overlap;
  LLM judge captures semantic quality beyond n-gram matching.
