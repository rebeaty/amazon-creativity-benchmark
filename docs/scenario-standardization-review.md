# Scenario Standardization Review

**Date:** 2026-02-24
**Scope:** All 122 scenario implementations in `scenarios/*/scenario.py`

---

## Summary

The scenario codebase has strong structural consistency at the HELM Scenario
level (class hierarchy, required attributes, instance format), but significant
divergence in internal implementation patterns. The biggest gaps are in dataset
loading, helper method naming, and multi-task handling. Additionally, the
RunSpec integration layer needed to actually execute scenarios as a benchmark
suite does not exist yet.

---

## 1. What's Well Standardized

These patterns are consistent across all 122 scenarios:

| Area | Standard | Compliance |
|------|----------|------------|
| Class hierarchy | All inherit from `Scenario`, implement `get_instances(self, output_path) -> List[Instance]` | 122/122 |
| Imports | `Scenario`, `Instance`, `Input`, `Output`, `Reference`, `CORRECT_TAG`, `TEST_SPLIT` | 122/122 |
| Class attributes | `name`, `description`, `tags` defined on every class | 122/122 |
| Reference tagging | `CORRECT_TAG` used universally for correct answers (no custom tag strings) | 122/122 |

## 2. What's Partially Standardized

### Docstrings

Most files include a class-level docstring, but lengths vary widely:

| Length | Count | Notes |
|--------|-------|-------|
| 20-50 lines | 100 | The de facto standard; most follow the onboarder template |
| >50 lines | 17 | Overly detailed (e.g., `creative_pair` at 73 lines) |
| <20 lines | 5 | Bare-minimum (e.g., `riddlesense` at 18 lines) |

### Error Handling

Only **11 of 122** scenarios include `try/except` blocks around dataset
loading. A few (e.g., `d_humor`, `creative_pair`) handle gated-dataset
authentication errors gracefully; the rest assume downloads succeed.

### Helper Method Naming

No consistent convention. Across scenarios, private methods use six different
prefixes:

| Prefix | Files using it | Purpose |
|--------|---------------|---------|
| `_download_*` | 32 | Fetch remote data |
| `_load_*` | 17 | Parse downloaded data |
| `_get_*` | 10 | Retrieve or compute sub-results |
| `_format_*` | 8 | Transform data for Instance creation |
| `_build_*` | 7 | Construct complex objects |
| `_create_*` | 6 | Instantiate HELM objects |

## 3. What's Not Standardized (Biggest Gaps)

### Multi-Task Handling

Scenarios that support multiple sub-tasks use three incompatible patterns:

| Pattern | Count | Example |
|---------|-------|---------|
| Single class, single task | 101 | Most scenarios |
| Single class with `task`/`subtask` param in `__init__` | 16 | `aaar` |
| Single class dispatching to `_get_*_instances()` helpers | 4 | `hummus` |
| Multiple `Scenario` classes in one file | 1 | `d_humor` |

### Dataset Loading

Seven different loading mechanisms are in use, with no common wrapper:

| Method | Count | Notes |
|--------|-------|-------|
| `json.load` / `json.loads` | 51 | Direct JSON parsing, often paired with `urllib` |
| `urllib` | 50 | Raw URL downloads |
| `load_dataset` (HuggingFace `datasets`) | 44 | Used for HF Hub-hosted data |
| `csv` module | 15 | Standard library CSV reader |
| `ensure_file_downloaded` (HELM utility) | 10 | HELM's built-in downloader |
| `pandas` | 9 | `pd.read_csv`, `pd.read_json`, etc. |
| `snapshot_download` (HuggingFace `huggingface_hub`) | 4 | For large/gated repos |
| `openpyxl` | 2 | Excel file parsing |

> Note: Many scenarios use multiple methods (e.g., `urllib` to download + `json.load` to parse), so counts overlap.

### `extra_data` Usage

Only **13 of 122** scenarios populate `Instance.extra_data` with metadata.
The rest leave it empty. There's no guidance on when or what to store there.

---

## 4. Module Registration

Only **4 of 122** scenarios have `__init__.py` files:

- `cs4`
- `fig_qa`
- `newyorker_humor`
- `vietnamese_poem`

The remaining 118 are not importable as Python packages, which means HELM's
module discovery cannot find them. This must be addressed before scenarios can
be executed.

---

## 5. Evaluation Configuration

Evaluation behavior is documented (not codified) via Markdown files alongside
scenarios:

| File | Count | Purpose |
|------|-------|---------|
| `annotator_notes.md` | 41 | LLM-as-judge rubrics, `RunSpec` examples |
| `metric_notes.md` | 32 | Custom metric requirements, scoring details |
| `evaluation_notes.md` | 1 | General evaluation guidance |
| **Total scenarios with any eval docs** | **69** | 57% coverage |

The remaining 53 scenarios have no evaluation configuration documentation.

Scoring methods from the benchmark catalog (283 tasks across all papers):

| Method | Count |
|--------|-------|
| Automatic (exact match, BLEU, etc.) | 116 |
| Mixed (automatic + human/LLM) | 106 |
| Human evaluation | 27 |
| LLM-as-judge | 21 |

---

## 6. Running the Full Benchmark

**The RunSpec layer does not exist yet.** The repository is in the "scenario
onboarding" phase. Each `scenario.py` is a standalone HELM data loader, but
there is no:

- `run_specs.conf` or `run_entries.conf`
- `pyproject.toml` / `setup.py` / `requirements.txt`
- Runner scripts or Makefiles
- Central scenario registry

### What's Needed

To run all scenarios as a HELM benchmark suite:

1. **Add `__init__.py` to all 118 remaining scenario directories** so HELM can
   discover them via Python's import system.

2. **Create a package manifest** (`pyproject.toml` or `setup.py`) that declares
   the `scenarios` package and its dependencies (HuggingFace `datasets`,
   `pandas`, `openpyxl`, etc.).

3. **Build a `run_specs.py`** (or equivalent config) that maps each scenario to
   its metrics. The `annotator_notes.md` and `metric_notes.md` files already
   sketch these out — they just need to be codified:

   | Scoring method | HELM metric spec |
   |----------------|------------------|
   | `exact_match` | `get_exact_match_metric_specs()` |
   | `open_ended` | `get_open_ended_generation_metric_specs()` |
   | `llm_judge` | Custom annotator + `RunSpec(...)` |
   | `custom` | New metric implementations |

4. **Execute via HELM CLI:**
   ```bash
   helm-run --run-entries "scenario_name:model=MODEL" --suite creativity-benchmark
   ```

---

## 7. Recommendations

| Priority | Action | Effort |
|----------|--------|--------|
| **P0** | Generate `__init__.py` for all 118 scenarios missing them | Low (scriptable) |
| **P0** | Create `pyproject.toml` with dependency list | Low |
| **P1** | Standardize helper method naming (`_download_*`, `_load_*`, `_format_*`) | Medium |
| **P1** | Add `try/except` with clear error messages to all dataset downloads | Medium |
| **P1** | Codify `annotator_notes.md` / `metric_notes.md` into executable RunSpecs | Medium |
| **P2** | Pick one multi-task pattern and migrate the 21 multi-task scenarios to it | Medium |
| **P2** | Document `extra_data` policy (what to store, when) | Low |
| **P2** | Write evaluation docs for the 53 scenarios currently missing them | High |
