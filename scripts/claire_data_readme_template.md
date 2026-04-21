# Creativity Benchmark — First Data Pack for Claire

**Date:** {DATE}
**Scope:** Two benchmark pairs (one close-ended, one open-ended) across all
available Gemini + Gemma model suites. This is a pilot for the stats /
IRT modeling pipeline before committing to the full ~90-benchmark rerun.

---

## What's in this folder

| File | Contents |
|---|---|
| `sampled_pair_long.tsv` | Long-format results, one row per (suite, model, evaluation_unit, item). Primary metric only. |
| `summary.txt` | Counts per (suite, model, unit) — sanity check for coverage gaps. |
| `README.md` | This file. |

## Columns in `sampled_pair_long.tsv`

| Column | Description |
|---|---|
| `suite` | Our internal run-suite name (e.g., `gemini_lite`). |
| `model` | Model API identifier (e.g., `google/gemini-2.5-flash-lite`). |
| `evaluation_unit` | `brainteaser_sampled:subtask=sentence_puzzle`, `brainteaser_sampled:subtask=word_puzzle`, `cs4_sampled:subtask=instruction`, `cs4_sampled:subtask=story`. |
| `item_id` | Stable within-unit item identifier (`id0`, `id1`, …). Same IDs appear for every model on the same unit. |
| `metric_name` | `exact_match` (brainteaser) or `llm_judge_creativity` (cs4). |
| `metric_value` | `exact_match`: 0.0 or 1.0. `llm_judge_creativity`: 1–5 (integer). |

Item IDs are shared across models within a unit, so you can pivot to
wide form for factor analysis.

## Benchmarks included

### `brainteaser_sampled` — close-ended, 4-way MCQ accuracy

- Paper: *BrainTeaser: Lateral Thinking Puzzles for LLMs* (Jiang et al., EMNLP 2023, [2310.05057](https://arxiv.org/abs/2310.05057)).
- Data: `tasksource/brainteasers` (HF).
- **Two subtasks, treated as separate evaluation units** (see sampling rule below):
  - `subtask=sentence_puzzle` (SP) — 396 items, sampled to 200.
  - `subtask=word_puzzle` (WP) — 396 items, sampled to 200.
- Metric: `exact_match` on the chosen MCQ letter (A–D).

### `cs4_sampled` — open-ended creative story generation

- Paper: *CS4: Measuring LLM Creativity by Controlling the Number of
  Story-Writing Constraints* (Lakkaraju et al., Oct 2024, [2410.04197](https://arxiv.org/abs/2410.04197)).
- Data: GitHub [anirudhlakkaraju/cs4_benchmark](https://github.com/anirudhlakkaraju/cs4_benchmark).
- **Two subtasks, treated as separate evaluation units**:
  - `subtask=instruction` — instruction-based (realistic fiction), 250 items, sampled to 200.
  - `subtask=story` — story-based (writing prompts), 250 items, sampled to 200.
- Metric: `llm_judge_creativity` — GPT-4o judge scores 1–5 on originality
  + narrative creativity + imaginative elements (rubric in the run spec).

## Sampling rule (apply to the full rerun)

```
For each benchmark:
  → Has internally-distinguishable subtasks?
      Yes → Each subtask ≥ 30 items AND independently scorable?
              Yes → Split into separate evaluation units
              No  → Pool; sample stratified by subtask
      No  → Single unit
  → Unit size ≥ 200 items?
      Yes → Random sample of 200 (stratified by subtask if pooled)
      No  → Use all items
```

- RNG seed: **20260421**, stored at `data/sampling/rng_seed.txt` in the repo.
- Per-unit seeds derive from `hash(RNG_SEED + unit_name) mod 2**31` — so
  samples are different across units but stable across reruns.
- Materialized sampled indices are in `data/sampling/<unit>_sample.json`.

## Known caveats

- **Judge variance.** `llm_judge_creativity` is a single-judge GPT-4o
  score at temperature 0. If you want judge-reliability estimates we
  can run a second judge (Claude Sonnet or Gemini) on the same items.
- **20-item smoke runs in the same suite dirs.** Earlier we ran these
  same 15 models at `MAX_INSTANCES=20` on the original (non-sampled)
  `brainteaser` and `cs4` benchmarks. Those dirs coexist in
  `benchmark_output/runs/<suite>/` alongside the new sampled runs; we
  only exported the sampled ones here. If you need the smoke data as a
  comparison, ping me.
- **Gemini 2.5 Pro `thinkingBudget=128`.** This model won't accept
  `thinkingBudget=0`, so it spends some output budget on internal
  reasoning. Most items fit in the 512-token output cap, but for cs4
  (long stories) it can truncate. If you see cs4 `llm_judge_creativity`
  looking systematically lower for `gemini-2.5-pro` it's partly this.
- **alpaca_eval_2 dropped.** Some earlier triage lumped AlpacaEval into
  the pool, but it's instruction-following, not creativity. Not in this
  pack. Will not be in the full rerun.

## Reproducibility

To regenerate this pack from the repo state:

```bash
bash eval_scripts/run_sampled_pair_all_suites.sh   # ~45 min, writes into benchmark_output/runs/<suite>/
python scripts/export_for_claire.py --output-dir claire_data_<DATE>
```

## Questions / things you might want

- Per-item prompts and model outputs (alongside the scores)? — easy add
  from `scenario_state.json`, let me know.
- Second judge for reliability? — swap the annotator in the run_spec,
  rerun cs4 at a fraction of cost.
- A non-sampled (all-items) version of either benchmark? — also easy.

— Roger (via the creativity-benchmark pipeline)
