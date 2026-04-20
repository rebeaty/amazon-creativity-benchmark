# Amazon Creativity Benchmark

A HELM-compatible evaluation suite for benchmarking the creative capabilities of language and vision-language models across ~90 creativity datasets (story generation, analogy, humor, puns, scientific hypothesis generation, image captioning, aesthetic judgment, and more).

---

## Overview

Every dataset in this benchmark is implemented as a HELM [Scenario](https://github.com/stanford-crfm/helm) plus a `RunSpec`, paired with an expected-metrics registry. A single orchestrator script evaluates your chosen model on every dataset in parallel, routes all model calls (including LLM-judge metrics) through [OpenRouter](https://openrouter.ai/), and writes per-dataset results under `benchmark_output/runs/first_full_trial/`.

Key files:

| Path | Purpose |
|---|---|
| [scenarios/](scenarios/) | One HELM `Scenario` per dataset |
| [run_specs/](run_specs/) | One `RunSpec` per dataset (metrics + annotators) |
| [eval_scripts/](eval_scripts/) | Per-dataset eval shell scripts + orchestrator |
| [data/registry/registry_metrics.yaml](data/registry/registry_metrics.yaml) | Expected metrics per dataset |
| [data/registry/registry_inference.yaml](data/registry/registry_inference.yaml) | Inference config per dataset |
| [data/list_dataset_1st_trial.json](data/list_dataset_1st_trial.json) | Datasets the orchestrator iterates over |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/anthropics/amazon-creativity-benchmark.git
cd amazon-creativity-benchmark
```

### 2. Create a Python 3.10 environment

```bash
conda create --name creativity-bench python=3.10 -y
conda activate creativity-bench
```

### 3. Install the package

```bash
pip install -e ".[eval,dev]"
```

This installs `crfm-helm>=0.5.12`, data-format libraries (Pillow, h5py, openpyxl, PyYAML), optional eval dependencies (diffusers, clip-score), and pytest.

Verify:

```bash
python -c "import helm; print(helm.__version__)"
```

### 4. Configure API keys

All model and judge calls are routed through OpenRouter. Create a local `.env` in the repo root (never commit it):

```bash
cat > .env << 'EOF'
export OPENROUTER_API_KEY="sk-or-..."
EOF
```

Get a key from https://openrouter.ai/keys. Make sure billing is enabled — OpenRouter is what gives you access to Claude, GPT, Gemini, Llama, and others through a single credential.

Load it into your shell:

```bash
source .env
echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY:0:10}..."
```

### 5. Quick sanity check

```bash
python -c "from scenarios import scenario_registry; print('scenarios OK')"
```

---

## Usage

The main entry point is [eval_scripts/00_run_all_parallel.sh](eval_scripts/00_run_all_parallel.sh). It verifies the target model and judge models are reachable on OpenRouter, then runs every dataset listed in `data/list_dataset_1st_trial.json` concurrently.

### Command

```bash
./eval_scripts/00_run_all_parallel.sh MODEL [MAX_INSTANCES] [PARALLELISM]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `MODEL` | yes | — | OpenRouter model slug (`vendor/model`) |
| `MAX_INSTANCES` | no | `-1` (all) | Cap on instances per dataset — useful for smoke tests |
| `PARALLELISM` | no | `4` | Number of datasets to run concurrently |

### Examples

```bash
# Full evaluation on all datasets, all instances, 4-way parallel
./eval_scripts/00_run_all_parallel.sh google/gemini-2.5-flash-lite

# Smoke test: 10 instances per dataset, 8-way parallel
./eval_scripts/00_run_all_parallel.sh anthropic/claude-sonnet-4 10 8

# Full eval with higher concurrency
./eval_scripts/00_run_all_parallel.sh openai/gpt-4o -1 16
```

### What happens when you run it

1. Loads `OPENROUTER_API_KEY` from `.env` (if present).
2. Fetches OpenRouter's model list and verifies `MODEL` + judge slugs (`openai/gpt-4-1106-preview`, `anthropic/claude-sonnet-4`, `google/gemini-2.5-flash-lite`) are all available. Aborts before spending any tokens if not.
3. Dispatches each dataset's `eval_scripts/<dataset>.sh` in the background, throttled to `PARALLELISM` workers at a time. HELM downloads any missing dataset data automatically on first run.
4. Writes per-dataset logs to `benchmark_output/runs/first_full_trial/_orchestrator_logs/<dataset>.log` and final results to `benchmark_output/runs/first_full_trial/<run_dir>/`.
5. Prints a summary with passed / failed / skipped counts and exits non-zero if anything failed.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | All datasets passed |
| 1 | One or more datasets failed or were skipped (missing per-dataset script) |
| 2 | Bad arguments, missing `OPENROUTER_API_KEY`, or OpenRouter list fetch failed |
| 3 | Target `MODEL` not available on OpenRouter |
| 4 | A required judge model is not available on OpenRouter |
| 5 | Dataset list file missing |
| 6 | Dataset list is empty |

### Running a single dataset

To debug or re-run one dataset in isolation:

```bash
bash eval_scripts/<dataset>.sh "$MODEL" first_full_trial ""
```

The third argument is `MAX_INSTANCES` (empty = all).

---

## License

MIT
