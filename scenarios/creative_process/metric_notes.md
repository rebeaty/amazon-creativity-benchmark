# Metric Notes: Creative Process

Source: Paper Section 3; scripts/make_csvs.ipynb and scripts/Analysis.ipynb in codebase

## Evaluation Framework

The paper's key contribution is **process analysis** — measuring HOW models
explore semantic space, not just WHAT they produce.

### Step 1: Sentence Embeddings
- Model: `thenlper/gte-large` (SentenceTransformers)
- Each generated word/phrase is embedded independently

### Step 2: Category Clustering
- Method: Hierarchical clustering on embeddings
- Distance metric: cosine distance
- Cutoff thresholds:
  - VF (animals): 1.14
  - AUT (brick/paperclip): task-specific
- Each response is assigned a category cluster ID

### Step 3: Semantic Similarity (SS)
- Cosine similarity between consecutive responses
- Range: [0, 1]
- Threshold: SS < 0.8 indicates a semantic "jump"

### Step 4: Jump Detection
- `jump_cat`: Adjacent responses in different category clusters
- `jump_SS`: Semantic similarity < 0.8
- `jump`: Combined (jump_cat AND jump_SS)

### Step 5: Jump Profile
- Cumulative count of jumps across the response sequence
- Characterizes exploration pattern:
  - **Persistent**: Deep exploration of few semantic spaces (fewer jumps)
  - **Flexible**: Broad exploration across many spaces (more jumps)

## Summary Metrics per Generation

| Metric | Description |
|--------|-------------|
| Total jumps | Number of semantic space transitions |
| Jump rate | Jumps / total responses |
| Unique categories | Number of distinct clusters explored |
| Mean SS | Average semantic similarity between consecutive items |
| Jump profile slope | Rate of semantic space exploration |

## Human Baselines

- 219 Dutch-speaking participants
- VF: ~30 animal names each (timed)
- AUT: Creative uses for brick (stimulus_fk=1) and paperclip (stimulus_fk=3)
- Data in repo: csvs/data_humans.csv (with English translations)

## LLM Baselines (Paper Table)

8 models tested at multiple temperatures (0.0-2.0), 5 reps each:
- Claude 3 Opus, GPT-4 Turbo, Gemini 1.0 Pro, Llama-3-70B,
  Mistral-7B, Nous-Hermes-2, PaLM text-bison, SOLAR-10.7B
