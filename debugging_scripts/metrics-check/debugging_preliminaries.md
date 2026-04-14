# Metrics-Check Debugging Sprint — Preliminaries

Follow these steps **exactly** in order. Do not skip steps.

> **Context**: This is **Phase 2** of debugging. Phase 1 (first-trial-run) confirmed
> that each dataset's eval script runs end-to-end and produces a `stats.json`.
> In this phase we verify that every **expected metric** (as defined in
> `data/registry/registry_metrics.yaml`) actually appears in the output. If
> metrics are missing, we diagnose and fix the run spec, scenario, or metric
> code until the output is complete.

---

## Assignees

| Name | Datasets |
|---|---|
| vijeta | 33 |
| clin | 33 |
| swastik | 33 |
| rajkumar | 32 |

Your assignment is in `debugging_scripts/metrics-check/debug_assignments_pending.json`.

---

## Step 0: GitHub Setup

### 0.1 Get the latest code

```bash
cd /home/public/vdeshpan/amazon-creativity-benchmark
git checkout master
git pull origin master
```

### 0.2 Configure your git identity (if not already done)

```bash
git config user.name "Your Full Name"
git config user.email "your.email@example.com"
```

### 0.3 Create your branch

Use your assigned name (lowercase): `vijeta`, `clin`, `swastik`, or `rajkumar`.

```bash
git checkout -b debug/metrics-check/<YOUR_NAME>
```

Examples:

```bash
git checkout -b debug/metrics-check/vijeta
git checkout -b debug/metrics-check/clin
git checkout -b debug/metrics-check/swastik
git checkout -b debug/metrics-check/rajkumar
```

Verify:

```bash
git branch
# Should show: * debug/metrics-check/<YOUR_NAME>
```

### 0.4 Push your branch to GitHub

```bash
git push -u origin debug/metrics-check/<YOUR_NAME>
```

---

## Step 1: Environment Setup

### 1.1 Create (or reuse) the conda environment

If you already have `creativity-bench` from Phase 1, activate it:

```bash
conda activate creativity-bench
```

Otherwise, create it fresh:

```bash
conda create --name creativity-bench python=3.10 -y
conda activate creativity-bench
```

### 1.2 Install dependencies

```bash
cd /home/public/vdeshpan/amazon-creativity-benchmark
pip install -e ".[eval,dev]"
```

### 1.3 Verify installation

```bash
python -c "import helm; print(helm.__version__)"
python -c "import yaml; print('PyYAML OK')"
```

---

## Step 2: API Keys

Same as Phase 1 — keys are loaded from a local `.env.local` file that is **never committed**.

### 2.1 Create or verify `.env.local`

```bash
cat /home/public/vdeshpan/amazon-creativity-benchmark/.env.local
```

If the file doesn't exist, create it:

```bash
cat > /home/public/vdeshpan/amazon-creativity-benchmark/.env.local << 'EOF'
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
EOF
```

Fill in your actual keys (see Phase 1 docs for where to get them).

### 2.2 Source the keys

```bash
source /home/public/vdeshpan/amazon-creativity-benchmark/.env.local
```

### 2.3 Verify

```bash
echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:0:20}..."
echo "OPENAI_API_KEY=${OPENAI_API_KEY:0:20}..."
echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY:0:20}..."
```

---

## Step 3: Understand the Workflow

### 3.1 What this phase checks

For each dataset, `metrics_check.py` compares two sets:

| Set | Source | Description |
|---|---|---|
| **m1** (expected) | `data/registry/registry_metrics.yaml` | Metrics the dataset *should* produce |
| **m2** (actual) | `outputs/first-trial-run/trial/<run_dir>/stats.json` | Metrics the eval *actually* produced |

A dataset **passes** when `m1 ⊆ m2` (every expected metric appears in the output).

### 3.2 What gets fixed when metrics are missing

| Root cause | Where to fix |
|---|---|
| Missing `MetricSpec` in run spec | `run_specs/<dataset>_run_specs.py` |
| Missing `AnnotatorSpec` for LLM-judge metric | `run_specs/<dataset>_run_specs.py` |
| Metric class doesn't exist in HELM | `metrics/` (create a custom adapter) |
| Scenario doesn't emit correct `Reference` fields | `scenarios/<dataset>_scenario.py` |
| Wrong metric name in registry | `data/registry/registry_metrics.yaml` (ask Vijeta first) |

### 3.3 Key files

| File | Purpose |
|---|---|
| `debugging_scripts/metrics-check/init_eval.sh` | Run eval for a single dataset (with auto-fix loop) |
| `debugging_scripts/metrics-check/run_metrics_check.sh` | Full orchestrator: check metrics, fix, re-eval, repeat |
| `debugging_scripts/metrics-check/metrics_check.py` | Compare expected vs actual metrics (exit 0/1/2) |
| `debugging_scripts/metrics-check/_debug_helper.py` | Bookkeeping: `next`, `done`, `status` |
| `debugging_scripts/metrics-check/debug_assignments_pending.json` | Your pending datasets |
| `debugging_scripts/metrics-check/debug_assignments_done.json` | Your completed datasets |
| `data/registry/registry_metrics.yaml` | Ground truth: expected metrics per dataset |

---

## Step 4: Verify Setup

### 4.1 Check your assignment

```bash
cd /home/public/vdeshpan/amazon-creativity-benchmark

# See how many datasets you have
python3 debugging_scripts/metrics-check/_debug_helper.py status <YOUR_NAME>

# List your pending datasets
python3 -c "
import json
with open('debugging_scripts/metrics-check/debug_assignments_pending.json') as f:
    pending = json.load(f)
for d in pending.get('<YOUR_NAME>', []):
    print(f'  - {d}')
"
```

### 4.2 Test the metrics checker on one dataset

Pick any dataset from your list and run:

```bash
python3 debugging_scripts/metrics-check/metrics_check.py <dataset> --format human
```

You should see output listing expected vs actual metrics and whether any are missing.

### 4.3 Test init_eval.sh (dry check)

Verify the eval script exists for one of your datasets:

```bash
ls eval_scripts/<dataset>.sh
```

---

## Step 5: Running the Debugging Scripts

### 5.1 Option A: Run a single dataset (recommended to start)

```bash
cd /home/public/vdeshpan/amazon-creativity-benchmark
conda activate creativity-bench
source .env.local

# Run the full metrics-check loop for one dataset
bash debugging_scripts/metrics-check/run_metrics_check.sh <YOUR_NAME> <dataset>
```

This will:
1. Check if all expected metrics are in `stats.json`
2. If yes: move the dataset to done
3. If no: call Claude Code to diagnose + fix, re-run eval, and retry (up to 10 times)

### 5.2 Option B: Run all your pending datasets

```bash
bash debugging_scripts/metrics-check/run_metrics_check.sh <YOUR_NAME>
```

This loops through your entire pending list.

### 5.3 Option C: Dry run (see what would run)

```bash
bash debugging_scripts/metrics-check/run_metrics_check.sh <YOUR_NAME> --dry-run
```

### 5.4 Run just the eval (without metrics check)

If you need to re-run the eval independently:

```bash
bash debugging_scripts/metrics-check/init_eval.sh <YOUR_NAME> <dataset>
```

---

## Step 6: Understanding the Output

### 6.1 Where output goes

All logs and learnings are saved per-assignee:

```
debugging_scripts/metrics-check/<YOUR_NAME>/
├── <dataset>_metrics_<timestamp>.log    # Full debug log from run_metrics_check.sh
├── <dataset>_<timestamp>.log            # Full debug log from init_eval.sh
├── <dataset>_diagnosis.md               # Claude's diagnosis of missing metrics
└── eval_run_learnings.md                # Accumulated learnings across all datasets
```

### 6.2 Exit codes

| Script | Code | Meaning |
|---|---|---|
| `metrics_check.py` | 0 | All expected metrics found |
| `metrics_check.py` | 1 | Some metrics missing |
| `metrics_check.py` | 2 | Dataset not in registry or no stats.json |
| `init_eval.sh` | 0 | Eval succeeded (stats.json written) |
| `init_eval.sh` | 1 | Eval failed after all attempts |
| `init_eval.sh` | 2 | Skipped (data access error) |

### 6.3 What happens after each dataset

| Outcome | What happens |
|---|---|
| All metrics present | Dataset moves from `pending` to `done` |
| Fixed and all metrics now present | Dataset moves from `pending` to `done` |
| Failed after 10 attempts | Dataset stays in `pending` — can be retried or escalated |
| Not in registry | Skipped — tell Vijeta so the registry can be updated |

---

## Step 7: Checking Your Progress

```bash
# Summary count
python3 debugging_scripts/metrics-check/_debug_helper.py status <YOUR_NAME>
# Output: <Name>: 5 done, 28 pending, 33 total

# List remaining datasets
python3 -c "
import json
with open('debugging_scripts/metrics-check/debug_assignments_pending.json') as f:
    pending = json.load(f)
for d in pending.get('<YOUR_NAME>', []):
    print(f'  - {d}')
"

# List completed datasets
python3 -c "
import json
with open('debugging_scripts/metrics-check/debug_assignments_done.json') as f:
    done = json.load(f)
for d in done.get('<YOUR_NAME>', []):
    print(f'  - {d}')
"
```

---

## Step 8: Git Workflow During Debugging

### 8.1 Before starting each day

```bash
git fetch origin
git merge origin/master
```

### 8.2 After fixing a dataset

```bash
git add run_specs/<dataset>_run_specs.py scenarios/<files_changed>
git commit -m "Metrics-check <dataset>: <brief description>"
```

Examples:

```bash
git commit -m "Metrics-check dat: added missing bleu MetricSpec to run spec"
git commit -m "Metrics-check macgyver: fixed AnnotatorSpec for judge_score_creativity"
```

### 8.3 Push your work

```bash
git push origin debug/metrics-check/<YOUR_NAME>
```

Do this at least once a day or after each batch of datasets.

### 8.4 Rules

- **Do NOT edit `data/registry/registry_metrics.yaml`** without asking Vijeta
- **Do NOT modify HELM's installed package files** (anything under `site-packages/`)
- Only modify: `run_specs/`, `scenarios/`, `metrics/`, `eval_scripts/`
- Your logs and learnings in `debugging_scripts/metrics-check/<YOUR_NAME>/` are yours — no conflict risk

---

## Step 9: Typical Workflow (Repeat Until Done)

```bash
# Daily startup
conda activate creativity-bench
source .env.local
cd /home/public/vdeshpan/amazon-creativity-benchmark
git fetch origin && git merge origin/master

# Run next dataset
bash debugging_scripts/metrics-check/run_metrics_check.sh <YOUR_NAME> <dataset>

# Commit and push
git add <changed_files>
git commit -m "Metrics-check <dataset>: <what changed>"
git push origin debug/metrics-check/<YOUR_NAME>
```

---

## Troubleshooting

### "metrics_check.py says 'no_registry' for my dataset"

The dataset is not listed in `data/registry/registry_metrics.yaml`. Tell Vijeta so
it can be added. Skip this dataset for now.

### "metrics_check.py says 'no_stats' — no stats.json found"

The Phase 1 eval output is missing. Run `init_eval.sh` to generate it:

```bash
bash debugging_scripts/metrics-check/init_eval.sh <YOUR_NAME> <dataset>
```

### "Claude keeps failing to fix the missing metric"

After 3-4 failed attempts, check the diagnosis file:

```bash
cat debugging_scripts/metrics-check/<YOUR_NAME>/<dataset>_diagnosis.md
```

Common issues:
- The metric requires a custom HELM class that doesn't exist yet (needs to be written in `metrics/`)
- The `AnnotatorSpec` points to a wrong annotator class
- The scenario doesn't emit `Reference` objects that the metric expects

If stuck, escalate to Vijeta with the log file.

### "init_eval.sh times out"

The default timeout is 120 seconds. Some datasets with large prompts or slow APIs
may need more time. Edit `init_eval.sh` line 23 (`EVAL_TIMEOUT=120`) to increase it.

### "I accidentally edited registry_metrics.yaml"

Revert it immediately:

```bash
git checkout -- data/registry/registry_metrics.yaml
```

### "merge conflict when pulling master"

```bash
# Open conflicted file, resolve by hand
git add <file>
git commit -m "Merge master into debug/metrics-check/<YOUR_NAME>"
git push origin debug/metrics-check/<YOUR_NAME>
```

Or ask Vijeta for help.

---

## File Reference

| File | Purpose |
|---|---|
| `debugging_scripts/metrics-check/debug_assignments_pending.json` | Datasets still to check (per person) |
| `debugging_scripts/metrics-check/debug_assignments_done.json` | Datasets completed (per person) |
| `debugging_scripts/metrics-check/_debug_helper.py` | Bookkeeping (`next`, `done`, `status`) |
| `debugging_scripts/metrics-check/metrics_check.py` | Expected-vs-actual metric comparison |
| `debugging_scripts/metrics-check/init_eval.sh` | Single-dataset eval with auto-fix |
| `debugging_scripts/metrics-check/run_metrics_check.sh` | Full orchestrator (check + fix + re-eval) |
| `debugging_scripts/metrics-check/<name>/` | Per-assignee logs, diagnoses, and learnings |
| `data/registry/registry_metrics.yaml` | Ground truth for expected metrics |

Check [CLAUDE.md](../../CLAUDE.md) for the full **Debugging Protocol for Each Dataset** checklist.
