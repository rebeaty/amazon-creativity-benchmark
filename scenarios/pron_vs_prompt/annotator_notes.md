# Annotator Notes: Pron vs Prompt — Creative Synopsis Generation

Source: arXiv:2407.01119; https://github.com/grmarco/pron-vs-prompt (data/rubric.json, data/expert_assessment.csv)

## Original Evaluation Method

Three expert literature critics evaluated each synopsis using a structured rubric
grounded in Boden's creativity dimensions (novelty, surprise, value). All criteria
scored 0–3 unless noted. No LLM judge was used in the original paper — expert humans
were the evaluators.

## LLM-as-Judge Configuration

**Judge model:** GPT-4o (recommended)
**Evaluation type:** Single-response literary quality assessment
**Scale:** 0–3 per dimension (matching expert rubric)

## Expert Rubric (from data/rubric.json)

### 1. Attractiveness (`atractivo`) — 3 sub-dimensions, 0–3 each

- **Title attractiveness**: Does the title attract your attention as a reader?
- **Style attractiveness**: Does the writing style attract you?
- **Story/character attractiveness**: Does the story or its characters attract you?

### 2. Originality (`originalidad`) — 3 sub-dimensions, 0–3 each

- **Title originality**: Is the title original?
- **Style originality**: Is the writing style original?
- **Plot originality**: Is the plot original?

### 3. Relevance (`relevancia`) — 1 question, 5-point scale (0–4)

How well did the writer use the given title as a creative starting point?
- 0 = The synopsis has nothing to do with the title
- 1 = The synopsis could fit any title
- 2 = The synopsis is inspired by the title
- 3 = The title is central to the story
- 4 = The title is used in a brilliant and original way

### 4. Creativity (`creatividad`) — 2 sub-dimensions, 0–3 each

- **Title creativity**: How creative is the use of the title overall?
- **Synopsis creativity**: How creative is the synopsis overall?

### 5. Literary criticism (`crítica`) — 3 binary questions (0/1)

- **Anthology inclusion**: Would you include this synopsis in an anthology of the best
  literary works of the year?
- **Reader agreement**: Would most readers agree with your literary assessment?
- **Critic agreement**: Would most critics agree with your literary assessment?
- **Recognizable voice**: Does the author have a recognizable literary voice?

## Judge Prompt Template

```
You are an expert literature critic. You will evaluate a creative story synopsis written
for an imaginary movie title. The synopsis should be approximately 600 words and have
literary value appealing to both critics and general audiences.

Title: {TITLE}
Synopsis: {RESPONSE}

Rate this synopsis on each of the following dimensions (0-3 unless noted):

Attractiveness:
- title_attractiveness (0-3): Does the title attract your attention as a reader?
- style_attractiveness (0-3): Does the writing style attract you?
- theme_attractiveness (0-3): Does the story or its characters attract you?

Originality:
- title_originality (0-3): Is the title original?
- style_originality (0-3): Is the writing style original?
- plot_originality (0-3): Is the plot original?

Relevance (0-4):
- relevance: How well did the writer use the given title as a creative starting point?
  (0=unrelated, 1=any title fits, 2=title inspired it, 3=title is central, 4=brilliant use)

Creativity:
- title_creativity (0-3): How creative is the use of the title?
- synopsis_creativity (0-3): How creative is the synopsis overall?

Literary quality (0 or 1):
- anthology (0/1): Would you include this in a literary anthology?
- own_voice (0/1): Does the author have a recognizable literary voice?

Return a JSON object with these keys and integer values.
```

## Human Calibration Data

Expert assessments are in `data/expert_assessment.csv` (GitHub repo).

Key columns:
| Column | Description |
|--------|-------------|
| `title` | Title text |
| `synopsis_writer` | Who wrote: `gpt4_en`, `patricio`, `claude`, etc. |
| `title_writer` | `patricio` or `machine` |
| `1_attractive_title/style/theme` | 0-3 attractiveness scores |
| `2_originality_title/style/theme` | 0-3 originality scores |
| `3_relevance` | 0-4 relevance score |
| `4_creativity_title/synopsis` | 0-3 creativity scores |
| `6_anthology`, `6_own_voice` | Binary literary quality |

Use for: calibrating LLM judge against 3 experts per synopsis; computing inter-rater
reliability; establishing human baseline scores.

## Paper Findings (for context)

- GPT-4 vs. Pron: experts found GPT-4 significantly below Pron on all dimensions
- GPT-4 is MORE creative when writing from Pron's titles than its own generated titles
- GPT-4 scores higher in English than Spanish
- Expert inter-rater agreement: moderate (creativity dimensions had higher variance)

## Notes

- Original evaluation used human expert critics, not LLM-as-judge
- 3 experts per synopsis across all 60 titles × 3 writers (Pron, GPT-4, Claude) conditions
- The English condition used `english_title` column; Spanish used `title` column
- `experiment` column in expert_assessment.csv distinguishes conditions (`ENGLISH` / main)
