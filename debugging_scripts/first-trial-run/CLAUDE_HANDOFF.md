# Handoff to Server-Side Claude

**You are picking this up on Vijeta's Linux server with zero prior context.**
A previous Claude Code session (Windows, April 2026) triaged + smoke-tested
the 59 pending benchmarks from `debug_assignments_pending.json` and pushed
results on this branch (`rbeaty/smoke-pending-2026-04-21`). Your job is to
run the full-scale evaluation of the 33 that passed smoke + resolve the
server-side-only blockers on the rest.

---

## Start here

1. **Read [SESSION_NOTES.md](SESSION_NOTES.md) end-to-end first.** It has
   per-dataset disposition, all code fixes with file:line references, and
   the categorized server-handoff list. Don't start work before you've
   read it — the categories are what drive your next actions.

2. **Verify your environment:**
   ```bash
   cd /home/public/vdeshpan/amazon-creativity-benchmark
   git fetch origin
   git checkout rbeaty/smoke-pending-2026-04-21
   conda activate creativity-bench
   source .env.local   # loads API keys
   python -c "import helm; print(helm)"
   ```

3. **Verify the code patches merged cleanly on Linux.** Five scenario /
   run_spec / metric-check fixes landed in commit `e2eb48b7`. They should
   apply unchanged on Linux (no Windows-specific shims). Sanity-check:
   ```bash
   git show e2eb48b7 --stat
   ```

4. **Re-verify smoke on one dataset before scaling.** The smoke runner
   lives at [smoke_all.sh](smoke_all.sh). It targets a `trial` suite with
   `google/gemini-2.5-flash-lite` at `max_instances=1`. On Linux the
   runtime will be faster than the Windows numbers in `_summary.tsv`.
   Just confirm the plumbing:
   ```bash
   bash eval_scripts/tinystories.sh google/gemini-2.5-flash-lite trial 1
   ```
   Expect `stats.json` in `benchmark_output/runs/trial/tinystories_model=google_gemini-2.5-flash-lite/`.

---

## What the previous Claude did (summary)

| Bucket | Count | Status |
|---|---|---|
| **33 RUNNABLE — ready for full-scale run** | 33 | 27 text-only + 6 multimodal (2 required code fixes; see commit e2eb48b7) |
| Server needs model swap (num_outputs>8 Google limit) | 3 | aidanbench, amuse_chord_generation, noveltybench |
| Server needs data staging (slow/large HF) | 4 | tiger_bench, ava, puzzleworld, vgsg |
| Server needs repo clone (SPLAT) | 1 | splat |
| Server needs scenario rewrite (no gold refs) | 1 | twistlist |
| Data access blocked (gated / gdrive / YouTube / EULA) | 15 | see SESSION_NOTES §Data-access skips |
| Dropped (duplicate) | 1 | graphragbench-wrongone |
| **Total** | **58** | ✓ (was 59, one drop) |

Note: `debug_assignments_pending.json` already has `graphragbench-wrongone`
removed on this branch. rajkumar's list is 31 now, not 32. Don't be
surprised.

---

## Your first pass

### Task A — run the 33 RUNNABLE at full instance count

SESSION_NOTES lists them by name. Launch per-assignee so the existing
pending-tracker logic continues to work:

```bash
bash debugging_scripts/first-trial-run/run_debug.sh Clin
bash debugging_scripts/first-trial-run/run_debug.sh Namrata
bash debugging_scripts/first-trial-run/run_debug.sh Sai
bash debugging_scripts/first-trial-run/run_debug.sh Vijeta
bash debugging_scripts/first-trial-run/run_debug.sh rajkumar
```

The orchestrator will skip the 25 non-runnable datasets on data-access
errors (that behavior is intentional — don't "fix" it).

### Task B — run the 3 `num_outputs>8` datasets under a non-Google model

Google's Gemini API caps `candidate_count` at 8. aidanbench uses
`num_outputs=30` by design (the task is "generate 30 distinct responses
to test breadth"); cannot be capped without breaking the task. Run under
OpenRouter:

```bash
MODEL=openai/gpt-5-mini   # or anthropic/claude-haiku-4.5, x-ai/grok-4.1-fast
for ds in aidanbench amuse_chord_generation noveltybench; do
  bash eval_scripts/${ds}.sh $MODEL trial 10
done
```

Verify `stats.json` lands. If HELM's AdapterSpec registration trips on a
non-Google deployment name, you may need to register an OpenRouter
deployment in `prod_env/model_deployments.yaml`.

### Task C — data staging (4 datasets)

- **tiger_bench** — HF dataset has 6000 files. On Windows it timed out
  at 360s downloading. Linux should be faster; if not, pre-stage via
  `huggingface-cli download`.
- **ava, puzzleworld, vgsg** — image HF datasets that timed out at 300s.
  Either raise timeout or pre-stage images.

### Task D — clone SPLAT

```bash
cd benchmark_output/scenarios/  # or wherever scenario expects
git clone https://github.com/<splat-repo>.git splat
```
The scenario hard-codes a local path; check
`scenarios/splat_scenario.py` for the expected location.

### Task E — twistlist scenario fix

`scenarios/twistlist_scenario.py` produces references with empty tags,
so `evaluate_reference_metrics.compute_reference_metrics` fails on
`assert len(golds) > 0`. You need to decide what the gold reference
should be for TwistList (source tweet? human-written twist?) and mark
it with `CORRECT_TAG` like the esp_dataset fix in this branch did
(commit e2eb48b7). This is a design decision, not a mechanical fix —
check the paper.

### Task F — 15 data-access blockers

SESSION_NOTES §Data-access skips has per-dataset blockers (HF-gated,
gdrive, YouTube, manual, EULA). Most need human action: HF access
requests, EULA signatures, or dataset author emails. Triage what's
worth the wait.

---

## Platform notes — Windows vs Linux

Things that were platform-specific and may NOT apply to you:

- **HELM `:` → `_` path patch.** The Windows filesystem rejects `:` in
  directory names, so a previous session monkey-patched `Runner._get_run_path`
  to replace `:` with `_`. On Linux you do NOT need this patch — run
  dirs should use the original `:` separator, matching HELM stock.
  If `metrics_check.py` starts mis-finding stats.json, that's why —
  it accepts both separators.

- **CRLF issues.** Some files may have Windows line endings. `git config
  core.autocrlf input` should handle conversion on checkout.

- **`hf_xet` was installed to speed up HF downloads on Windows.** On
  Linux you probably don't need it; HF's default is fine.

- **Gemini `thinkingBudget: 0`** was set in `model_deployments.yaml`
  for the 2.5 Flash models because their default thinking mode was
  eating the 512-token output budget on creative writing tasks. For
  Gemini 2.5 Pro the API rejects `thinkingBudget: 0` — had to use `128`.
  Watch for this if you switch default models in that file.

---

## Things the previous Claude didn't do (so you can pick up)

- **Classifier full run is on hold.** Roger plans to use EFA
  (exploratory factor analysis on the per-benchmark model scores) to
  find the number of domains empirically, then label them top-down with
  a theory-anchored taxonomy. The script
  `ai/review/classify_benchmarks_v3.py` (in the other repo —
  `OneDrive.../Amazon Benchmark/ai/review/`) was rewritten with v2
  paper_id linking + pruned theory block + multi-provider slate, and
  passed a 20-benchmark smoke test. **Do not run the full N=10 version
  unless Roger asks.** He wants to let EFA drive domain cardinality
  first.

- **Tiger_bench + twistlist + splat are not "someone else's problem"
  excuses to skip.** They're real benchmarks worth keeping in the corpus.
  Please put in the effort.

- **The 15 data-access-blocked datasets have varying recoverability.**
  Some just need someone to click "request access" on HF and wait 3
  days. Others (muse_perception) require signing an EULA. Some
  (webnovelbench) need author contact. Worth treating as a parallel
  workstream while Task A runs.

---

## How to report back

Keep the per-dataset fix log in SESSION_NOTES.md format if you add to it.
Append a new `## Server-side continuation — YYYY-MM-DD` section rather
than overwriting the Windows session's notes. That way Roger can
reconstruct the history.

Good luck. —  Previous Claude (Windows session, April 2026)
