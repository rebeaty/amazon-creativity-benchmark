# Annotator Notes: TheNextChapter — LLM-as-Judge Evaluation

Source: https://github.com/ZhuohanX/TheNextChapter/tree/main/HumanEvaluation
        Paper: "The Next Chapter" (INLG 2023), Section 4 — Human Evaluation

## Task

Given a story condition (prompt) and a model-generated story continuation,
score the story on five quality dimensions using a 1–5 scale.

## Configuration

Judge model: GPT-4 (or equivalent)
Reference available: Yes — human-written gold story in `reference` field
Instances available with pre-scored examples: 20 per subset × 3 subsets × 2
  annotation sets (crowdsource + inhouse) = 120 pre-scored instances

## Scoring Dimensions (1–5 scale each)

| Dimension | Description |
|-----------|-------------|
| `fluency` | Grammatical correctness, readability, and natural language flow |
| `coherence` | Internal consistency; events and characters remain logical throughout |
| `relatedness` | How well the story follows from and relates to the given condition |
| `logicality` | Plausibility of events; cause-and-effect relationships make sense |
| `interestingness` | Creativity, engagement, and entertainment value of the story |

## Judge Prompt Template

Score the following story continuation on each of the five dimensions below.
Use a scale of 1 (very poor) to 5 (excellent).

Condition (story prompt):
{condition}

Story continuation:
{RESPONSE}

Rate each dimension:
1. Fluency (1-5): How grammatically correct and naturally readable is the story?
2. Coherence (1-5): How internally consistent is the story?
3. Relatedness (1-5): How well does the story follow from the condition?
4. Logicality (1-5): How plausible are the events in the story?
5. Interestingness (1-5): How creative and engaging is the story?

Provide your rating for each dimension as a single number, then a brief
justification.

## Calibration

Pre-scored examples are available in HumanEvaluation/inhouse/ and
HumanEvaluation/crowdsource/ (20 items each for roc, wp, cnn subsets).
These can be used as few-shot examples to calibrate the judge.

## Notes

- Paper found GPT-3 (Davinci) outperformed all fine-tuned models on
  interestingness and relatedness; PROGEN3 was competitive on fluency/coherence.
- Human evaluation was conducted both in-house (researchers) and via
  crowdsourcing (MTurk); results were broadly consistent.
- The `roc` split uses gender placeholders ([MALE], [FEMALE], [NEUTRAL]);
  judge should interpret these as generic character references.
