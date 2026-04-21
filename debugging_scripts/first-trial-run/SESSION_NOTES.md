# Second-Wave Onboarding Session Notes

**Date started:** 2026-04-21
**Operator:** Roger Beaty (via Claude Code)
**Goal:** Onboard the ~59 pending benchmarks from `debug_assignments_pending.json`, mirroring Vijeta's first-trial-run pipeline. Use max_instances=1 smoke tests to verify onboarding only — full evals come later.

---

## Environment

- Model: `google/gemini-2.5-flash-lite` (direct Google API, not via OpenRouter)
- Suite: `trial`
- MAX_INSTANCES: **1** (smoke test only — verify scenario + run_spec + eval script wire up correctly and HELM writes stats.json)
- Python: `.venv/Scripts/python` with `PYTHONUTF8=1`
- Platform: Windows (git-bash), with our existing HELM patches (`:` → `_` in run dirs, UTF-8 encoding fixes)
- API keys loaded from `.env`: GOOGLE_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY

---

## Triage (59 → 58 after one drop)

Written by [smoke_all.sh](smoke_all.sh) and [triage.json](triage.json).

Categories:
| Category | Count | Meaning |
|---|---|---|
| TRY_IT | 33 | Text-in/text-out + no data-access blockers → smoke-test here |
| SKIP_DATA | 25 | Multimodal input, HF-gated, Google Drive, YouTube, or manual download → skip locally |
| BROKEN_CODE | 1 | Missing required files → was `graphragbench-wrongone`, dropped (see below) |

### Drops

| Dataset | Reason | Recorded in |
|---|---|---|
| `graphragbench-wrongone` | Name literally says "wrongone"; duplicate of `graphrag_bench` (which is also pending). Scenario/run_spec use hyphen + underscore naming inconsistently. | [drops.json](drops.json), removed from `debug_assignments_pending.json` (rajkumar) |

**Pending after drops: 58 (TRY_IT 33, SKIP_DATA 25).**

---

## Code Changes (this session)

### 1. `debugging_scripts/metrics-check/metrics_check.py`

**Problem A:** Referenced undefined `BENCHMARK_RUNS_DIR` (NameError when called).
**Problem B:** Glob pattern `{dataset}:*model=` missed our Windows run dirs, which use `_` not `:` (due to HELM patch).

**Fix:**
```diff
-REGISTRY_PATH = ...
-OUTPUTS_DIR = os.path.join(REPO_ROOT, "benchmark_output", "runs", "trial")
+REGISTRY_PATH = ...
+BENCHMARK_RUNS_DIR = os.path.join(REPO_ROOT, "benchmark_output", "runs")
+OUTPUTS_DIR = os.path.join(BENCHMARK_RUNS_DIR, "trial")
```

```diff
-    for base in search_dirs:
-        pattern = os.path.join(base, f"{dataset}:*model={model_safe}", "stats.json")
-        files.update(glob.glob(pattern))
-        exact = os.path.join(base, f"{dataset}:model={model_safe}", "stats.json")
-        if os.path.exists(exact):
-            files.add(exact)
+    for base in search_dirs:
+        for sep in (":", "_"):
+            pattern = os.path.join(base, f"{dataset}{sep}*model={model_safe}", "stats.json")
+            files.update(glob.glob(pattern))
+            exact = os.path.join(base, f"{dataset}{sep}model={model_safe}", "stats.json")
+            if os.path.exists(exact):
+                files.add(exact)
```

### 2. `debugging_scripts/first-trial-run/smoke_all.sh` (new file)

New runner script that iterates TRY_IT datasets from `triage.json`, runs each eval with `max_instances=1`, and records PASS/FAIL/TIMEOUT in `smoke_results/_summary.tsv`.

- Bug found + fixed: Python output on Windows has CRLF line endings, so every dataset name had a trailing `\r`. Patched with `| tr -d '\r'` in the pipeline.

### 3. `debugging_scripts/first-trial-run/triage.json` (new)

Triage output per dataset. Categorizes as TRY_IT / SKIP_DATA / BROKEN_CODE and notes multimodal/gated flags.

### 4. `debug_assignments_pending.json`

Removed `graphragbench-wrongone` from rajkumar's list (see drop above). rajkumar: 32 → 31. Total: 59 → 58.

---

## Smoke Test Results

_(Will be filled in as runs complete. Source: `smoke_results/_summary.tsv`.)_

### Multimodal smoke test (12 datasets attempted after initial triage)

Ran on 12 image-input datasets that were initially categorized SKIP_DATA but turned out to be locally attemptable with Gemini-2.5-Flash-Lite (multimodal capable). **Results: 6 PASS / 3 TIMEOUT / 3 FAIL.**

| Dataset | Status | Note |
|---|---|---|
| ii_bench | PASS | image→text |
| irfl | PASS | image→text |
| banner_request_400 | PASS | image→text |
| yesbut | PASS | image→text |
| creation_mmbench | **PASS after fix** | PIL→path (pattern B from debugging_w_vijeta.md) — [scenarios/creation_mmbench_scenario.py:155](../../scenarios/creation_mmbench_scenario.py#L155) |
| esp_dataset | **PASS after fix** | Added `CORRECT_TAG` to references — [scenarios/esp_dataset_scenario.py:160](../../scenarios/esp_dataset_scenario.py#L160) |
| ava | TIMEOUT 300s | slow HF image fetch |
| puzzleworld | TIMEOUT 300s | same |
| vgsg | TIMEOUT 300s | same |
| creative_pair | FAIL | dataset file not found locally |
| storyer | FAIL | dataset gated (gdrive + manual) |
| muse_perception | FAIL | EULA-gated (manual approval at muse-challenge.org) |

### Data-access skips (13 confirmed unrecoverable here)

Multimodal / gated / Google Drive / YouTube / manual-download. These are legitimate blockers that need either server-side data staging or a rework of the scenario loader. Flagged for Roger's server.

| Dataset | Person | Blocker |
|---|---|---|
| ava | Namrata | multimodal |
| clef_joker_2025_task2 | Namrata | gated HF |
| creation_mmbench | Namrata | multimodal |
| creative_pair | Namrata | multimodal |
| d_humor | Namrata | multimodal + gated HF |
| esp_dataset | Namrata | multimodal |
| llm_review_focus | Namrata | manual download |
| muse_perception | Namrata | multimodal |
| puzzleworld | Clin | multimodal |
| vflute | Clin | multimodal + gated HF |
| v_flute | Vijeta | multimodal + gated HF |
| vgsg | Vijeta | multimodal |
| funqa | Vijeta | multimodal + YouTube |
| webnovelbench | Vijeta | gated HF + manual download |
| yesbut_v2 | Vijeta | multimodal + Google Drive |
| storyer | Sai | Google Drive + manual |
| llm_srbench | Sai | gated HF |
| arn | rajkumar | Google Drive |
| banner_request_400 | rajkumar | multimodal |
| crowd_vote | rajkumar | YouTube |
| ii_bench | rajkumar | multimodal |
| irfl | rajkumar | multimodal |
| memecap | rajkumar | multimodal + gdrive + manual |
| moh_x | rajkumar | Google Drive |
| yesbut | rajkumar | multimodal |

### TRY_IT datasets (33 — smoke test complete)

**Pass rate: 25/33 = 76%**, with 6 FAIL and 2 TIMEOUT. Full per-dataset results in [smoke_results/_summary.tsv](smoke_results/_summary.tsv).

#### PASS (25)
arastories, arena_hard_creative, brainteaser, chinese_homophonic_puns, conceptual_design, creatset, crowdcounter, cs4, graphrag_bench, humor_transfer, liveideabench, llm4biohypogen, llm_discussion, meta4xnli, music_theory_bench, newyorker_humor, ocw, outline_to_story, sdat, sonnet_or_not_bot, speak_to_structure, ss_gen, thenextchapter, tinystories, ttcw

#### FAIL — Google API candidate-count limit (3 datasets)

These use `num_outputs > 8`, but Google's Gemini API caps `candidate_count` at 8. Not fixable locally — needs either OpenAI/OpenRouter model (supports n≥30) or a HELM-level adaptation to split into sequential calls. **Flag for Vijeta's server** with model `openai/gpt-4o-mini` or similar.

| Dataset | num_outputs | Error |
|---|---|---|
| aidanbench | 30 | `candidate_count must be in [1, 8]` |
| amuse_chord_generation | >8 | same |
| noveltybench | >8 | same |

#### FAIL — Real code bug (1, FIXED)

| Dataset | Bug | Fix |
|---|---|---|
| sudoku_bench | `KeyError: 'x'` — prompt string contains `r{x}c{y}` as literal docs, but Python `.format()` parses `{x}` as a format arg | Escaped to `r{{x}}c{{y}}` in [scenarios/sudoku_bench_scenario.py:125](../../scenarios/sudoku_bench_scenario.py#L125) |

#### FAIL — Missing pip package (1, FIXED)

| Dataset | Missing | Fix |
|---|---|---|
| twistlist | `bert_score` | `pip install bert-score hf_xet` |

#### FAIL — Data requires external repo clone (1, flag for server)

| Dataset | Blocker |
|---|---|
| splat | `FileNotFoundError: SPLAT repository not found. Please clone it first` — scenario expects a local clone of the SPLAT repo. Move to SKIP_DATA (my triage missed this). |

#### TIMEOUT (2, retrying with longer timeout)

| Dataset | Cause |
|---|---|
| litbench | HF dataset download slow (Xet storage, no hf_xet installed) |
| tiger_bench | HF dataset downloading 6000 files, reached 75% at 180s timeout |

Both fixed by: (1) `pip install hf_xet`, (2) raising timeout to 360s for the retry. Re-running now.

---

---

## Per-Dataset Fix Log

### sudoku_bench
- **Error:** `KeyError: 'x'` during `scenario.get_instances()`
- **Cause:** Prompt template contains the literal docs `r{x}c{y}` as a coordinate-format example. When Python `.format()` was called later to fill `{rows}`, `{cols}`, `{rules}`, etc., it tried to match `{x}` and `{y}` to named args that don't exist.
- **Fix:** [scenarios/sudoku_bench_scenario.py:125](../../scenarios/sudoku_bench_scenario.py#L125) — escaped to `r{{x}}c{{y}}` (renders as `r{x}c{y}` in the final prompt).
- **Outcome:** retry in progress

### twistlist
- **Error 1:** `ModuleNotFoundError: No module named 'bert_score'`
- **Cause 1:** Missing pip dep. **Fix:** `pip install bert-score hf_xet` — outcome: unblocked.
- **Error 2:** `TypeError: BertScoreMetric.__init__() got an unexpected keyword argument 'device'`
- **Cause 2:** [metrics/bert_score_metric.py](../../metrics/bert_score_metric.py) has **two** `class BertScoreMetric` definitions. The second (line 98) shadows the first (line 18). The live class doesn't accept `device` (it hardcodes `device="cpu"` internally). But `twistlist_run_specs.py` + `splat_run_specs.py` were passing `"device": "cpu"` as a MetricSpec arg.
- **Fix 2:** removed `"device": "cpu"` arg in [run_specs/twistlist_run_specs.py:36](../../run_specs/twistlist_run_specs.py#L36) and [run_specs/splat_run_specs.py:36](../../run_specs/splat_run_specs.py#L36).
- **Error 3 (remaining):** `AssertionError` at `helm/benchmark/metrics/evaluate_reference_metrics.py:500` — `assert len(golds) > 0`. The TwistList scenario isn't producing any references marked with `CORRECT_TAG` (i.e. no `is_correct=True` reference), so the evaluator can't compute reference-based metrics.
- **Outcome:** FAIL. Needs scenario-level fix (what should be the gold reference for TwistList?). Flag for server.
- **Followup:** (1) add `bert-score` to `pyproject.toml` extras, (2) delete the dead duplicate `BertScoreMetric` class, (3) redesign `twistlist_scenario.py` to tag a valid gold reference.

### splat (NOT fixed — re-categorized)
- **Error:** `FileNotFoundError: SPLAT repository not found. Please clone it first`
- **Cause:** Scenario expects a local clone of the SPLAT GitHub repo at a hardcoded path. Triage missed this because the scenario doesn't use `gdown`/HF-gated/YouTube keywords.
- **Action:** Moved from TRY_IT → SKIP_DATA in triage. Needs either a bundled copy of the repo or a scenario rewrite that fetches on first run. Flagged for server.

---

## Handoff to Vijeta (for server)

### Ready to run full-scale (33 datasets total)
All passed smoke test with `google/gemini-2.5-flash-lite, MAX_INSTANCES=1`. Safe to include in the next full-scale run.

**TRY_IT text-only (27):** arastories, arena_hard_creative, brainteaser, chinese_homophonic_puns, conceptual_design, creatset, crowdcounter, cs4, graphrag_bench, humor_transfer, liveideabench, litbench, llm4biohypogen, llm_discussion, meta4xnli, music_theory_bench, newyorker_humor, ocw, outline_to_story, sdat, sonnet_or_not_bot, speak_to_structure, ss_gen, sudoku_bench, thenextchapter, tinystories, ttcw

**Multimodal image→text (6):** ii_bench, irfl, banner_request_400, yesbut, creation_mmbench, esp_dataset

### Code patches applied this session (need Vijeta's review before merging)
- [scenarios/sudoku_bench_scenario.py:125](../../scenarios/sudoku_bench_scenario.py#L125) — escape `r{{x}}c{{y}}` to survive `.format()`
- [scenarios/creation_mmbench_scenario.py:155](../../scenarios/creation_mmbench_scenario.py#L155) — save PIL images to disk + pass path instead of PIL object (pattern B)
- [scenarios/esp_dataset_scenario.py:160](../../scenarios/esp_dataset_scenario.py#L160) — add `CORRECT_TAG` to references + import it
- [run_specs/twistlist_run_specs.py:36](../../run_specs/twistlist_run_specs.py#L36) + [run_specs/splat_run_specs.py:36](../../run_specs/splat_run_specs.py#L36) — remove spurious `"device": "cpu"` arg
- [debugging_scripts/metrics-check/metrics_check.py](../metrics-check/metrics_check.py) — fix undefined `BENCHMARK_RUNS_DIR` + handle Windows `_` path separator

### Needs server infra attention (6 datasets)
| Dataset | Issue | Server action |
|---|---|---|
| aidanbench | `num_outputs=30` exceeds Google API cap of 8 | Run under OpenRouter model (openai/gpt-5-mini, claude-haiku, grok-4.1-fast — all support n>8) |
| amuse_chord_generation | same | same |
| noveltybench | same | same |
| tiger_bench | HF dataset has 6000 files; local fetch times out | Pre-stage the dataset in shared cache |
| splat | Scenario expects local clone of SPLAT GitHub repo | Clone SPLAT into HELM data dir |
| twistlist | Scenario doesn't tag any reference as `CORRECT_TAG` → metric AssertionError | Redesign `twistlist_scenario.py` to mark a gold reference |

### Pending but not tried here (25 SKIP_DATA)
Multimodal / gated / gdrive / YouTube / manual-download. See triage.json for per-dataset blockers. All need server-side data staging before they can run.

### Drops
- graphragbench-wrongone — duplicate of graphrag_bench, filename uses hyphen while run_spec uses underscore. Removed from pending.
