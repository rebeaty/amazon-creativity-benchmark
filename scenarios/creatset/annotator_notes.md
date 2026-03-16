# Annotator Notes: CreataSet — Chinese Creative Writing

Source: arXiv:2505.19236; https://github.com/Aman-4-Real/CrEval

## Original Evaluation Method

The paper uses **pairwise creativity comparison** via human annotators and CrEval
(a trained LLM evaluator). Each response is ranked against the others using
Bradley-Terry pairwise comparisons (30 comparisons per response per instance).

## LLM-as-Judge Configuration

**Recommended judge:** GPT-4o or CrEval-7b (Aman/CrEval-7b on HuggingFace)
**Framework:** Pairwise comparison — which of two responses is more creative?
**Language:** Chinese (instructions and responses are in Chinese)

## Judge Prompt Template

From the training data instruction field (train_300k.json), verbatim:

```
你是一个专业的文本创意性评估专家，非常擅长判断对同一题目的两种不同回答中哪一个更具有创意性，
在你做出最终判断时，你需要考虑以下几个方面：
1. 原创性：回答是否具有独特的视角或想法，而不是简单地重复已知的信息。
2. 出乎意料性：回答是否能够给读者带来惊喜或新鲜感，超越常规思维。
3. 价值性：回答是否有意义、有深度，能够给读者带来启发或思考。

你必须始终输出"更有创意的回复是：Response X"（X为1或2），不要输出其他内容！

[指令]
{INSTRUCTION}

[Response 1]
{RESPONSE_1}

[Response 2]
{RESPONSE_2}

更有创意的回复是：
```

**English translation of judge criteria:**
1. Originality: Does the response have a unique perspective rather than repeating known info?
2. Unexpectedness: Does it surprise the reader or go beyond conventional thinking?
3. Value: Is it meaningful, deep, and inspiring to the reader?

**Output format:** "更有创意的回复是：Response X" (X = 1 or 2)

## Human Calibration Data

Each instance in `CreataSet-test_with_labeling_400.jsonl` includes:

| Field | Description |
|-------|-------------|
| `avg_score` | 5 Bradley-Terry win-rate scores for 5 candidate responses |
| `labeling` | 5×30 matrix of human pairwise comparison results |
| `gen_resp_order` | Names of 4 models: [MiniCPM-2B-c, Qwen2.5-14B-c, GPT4o-mini-c, GPT4o-mini-n] |

**Response index mapping:**
- Index 0: MiniCPM-2B-c (avg ~2.7)
- Index 1: Qwen2.5-14B-c (avg ~1.9)
- Index 2: GPT4o-mini-c (avg ~3.0, typically highest)
- Index 3: GPT4o-mini-n (avg ~2.4)
- Index 4: `output` field / unlisted reference (avg ~1.6, typically lowest)

Use `avg_score` to calibrate judge ratings against human pairwise preferences.
GPT4o-mini-c consistently receives the highest human creativity ratings.

## Evaluation Protocol

To evaluate a new model's output:
1. Generate response R for each instruction
2. Compare R pairwise against `gen_resp_2` (GPT4o-mini-c, the strongest baseline)
   using the judge prompt above
3. Win rate vs. GPT4o-mini-c is the primary metric
4. Optionally compare against all 4 reference model responses for full ranking

## Domain Coverage

| Domain | Count | Description |
|--------|-------|-------------|
| Short Texts | 50 | Brief creative Chinese text (poems, sayings, captions) |
| Lyrics | 50 | Chinese song lyrics |
| Modern Poetry | 50 | Contemporary Chinese poetry |
| Ancient Poetry | 50 | Classical Chinese poetry |
| Prose | 50 | Chinese prose/essay writing |
| RuoZhiBa | 50 | Zhihu-style creative Q&A (Chinese platform) |
| Oorigi-Go | 50 | Original creative writing prompts |
| Infinity-Instruct | 50 | Instruction-following creative tasks |

## Notes

- All instructions and responses are in Chinese (Simplified)
- CrEval-7b (Aman/CrEval-7b) is the paper's trained evaluator; requires LLaMA-Factory
- For simpler evaluation, use GPT-4o with the verbatim judge prompt above
- The pairwise comparison task (test_paired_3196.jsonl, 3,196 items) tests LLM ability
  to judge creativity, not generate it — skipped in this scenario
- Dataset loads via hf_hub_download (not load_dataset) due to schema mismatch between
  the two test files in the same split
