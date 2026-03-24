---
name: generate-metrics
description: >
  Generate HELM-compatible Python metric classes for missing metrics in the Amazon Creativity Benchmark.
  Use this skill when asked to: create metric implementations, fill in missing metrics, generate HELM-adaptable
  metric code, update the metric registry, or work with the creativity benchmark's metric pipeline. Also trigger
  when the user mentions distinct_1, distinct_2, JSD, MAE, correlation metrics, group_match_score,
  classification_accuracy, validity checks, pass_at_1, creativity_score, type_token_ratio, or any metric
  listed in the registry as having a null HELM class.
---

# Generate HELM-Compatible Metric Classes

## Overview

This skill reads a metric registry, identifies metrics with no existing HELM class (`null`), and generates
a Python file for each one. Every generated class inherits from the correct HELM base class and implements
the required `evaluate_generation` method so it can plug directly into the HELM evaluation harness.

---

## Fixed Paths — Use Exactly As Written

| Item | Path |
|------|------|
| Project root | `/home/public/vdeshpan/amazon-creativity-benchmark` |
| Metric registry | `{root}/data/registry/registry_per_metric_breakdown.md` |
| Output directory | `{root}/metrics` |
| HELM metrics (reference) | `https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/metrics` |

> **CRITICAL:** Never shorten, guess, or invent paths. Always use these exact paths.

---

## Execution Workflow

Follow these steps **in order**. Do not skip ahead.

### Step 0 — Install Dependencies

```bash
pip install --break-system-packages numpy scipy scikit-learn lxml
```

Spacy is needed only for `type_token_ratio`. Install it lazily in that metric's file rather than globally,
because the required model download (`en_core_web_sm`) may fail without network.

### Step 1 — Read the Registry

Open `{root}/data/registry/registry_per_metric_breakdown.md` and parse it.
Identify every row where the HELM Class column is `null`. These are the missing metrics you must generate.

Do **not** rely on a hardcoded list — always read the file fresh so the skill stays correct if the registry changes.

### Step 2 — Generate One Python File Per Metric

For each missing metric, create a file at:

```
{root}/metrics/{metric_name}_metric.py
```

For example, `distinct_1` → `distinct_1_metric.py`.

**Exception — grouped metrics:** When two or more metrics share identical logic differing only by a parameter
(e.g., `distinct_1` / `distinct_2`, `jensen_shannon_divergence_unigram` / `jensen_shannon_divergence_bigram`,
`pearson_correlation` / `spearman_correlation`), put them in **one file** named after the group:

| File | Metrics inside |
|------|---------------|
| `distinct_n_metric.py` | distinct_1, distinct_2 |
| `jsd_metric.py` | jensen_shannon_divergence_unigram, jensen_shannon_divergence_bigram |
| `correlation_metric.py` | pearson_correlation, spearman_correlation |
| `vietnamese_poem_metric.py` | poem_score, length_score, tone_score, rhyme_score |

All other metrics get their own file.

### Step 3 — Choose the Right Base Class

Almost every metric here scores one instance at a time, so **default to inheriting from `Metric`**.

Use `EvaluateInstancesMetric` only when the metric fundamentally requires cross-instance computation
(e.g., dataset-level bias scores). None of the current missing metrics require this — they all operate
per-instance. If a future metric does require it, the decision rule is:

- Can you score a single instance without seeing any other instance? → **`Metric`**
- Must you see all instances to compute the score? → **`EvaluateInstancesMetric`**

### Step 4 — Validate Each Generated File

After writing each file, run:

```bash
python -c "import ast; ast.parse(open('{file_path}').read()); print('OK')"
```

If syntax validation fails, fix the file before moving on.

### Step 5 — Update the Registry

Open `{root}/data/registry/registry_per_metric_breakdown.md`.
For each metric you generated, change its HELM Class cell from `null` to the class name you created
(e.g., `DistinctNMetric`). Do **not** alter any rows you did not generate code for.

### Step 6 — Rebuild the Summary (If Applicable)

If a separate metric summary file exists at the same path or a sibling path, regenerate it to reflect
the updated class mappings. If the summary is the same file as the registry, the Step 5 edit is sufficient.

---

## Class Template

Every generated metric must follow this structure:

```python
from typing import List
from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

class MyMetric(Metric):
    """One-line description of what this metric measures."""

    def __init__(self, some_param: str):
        super().__init__()
        self.some_param = some_param

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        # 1. Extract model output
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        # 2. Extract ground-truth references
        references = request_state.instance.references
        # If a single reference string is expected:
        # reference_text = references[0].output.text.strip()

        # 3. Compute score
        score = ...

        return [Stat(MetricName("my_metric")).add(score)]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="my_metric_score",
                display_name="My Metric Score",
                short_display_name="Score",
                description="Description of what this metric measures.",
                lower_is_better=False,
                group="accuracy",
            ),
        ]
```

**Rules:**
- Always call `super().__init__()` if you override `__init__`.
- Stat names must exactly match the metric name in the registry (e.g., `"distinct_1"`).
- Use `.add(float_value)` — never pass None.
- If computation fails (empty text, parse error), return a score of `0.0` rather than raising.
- Keep third-party imports inside the method body or behind a try/except if the library may be absent.

---

## Per-Metric Implementation Guide

Each section below tells you **exactly** how to compute the metric inside `evaluate_generation`.
The completion text is `completion`; the reference text (when needed) is `reference_text`.

### distinct_1 and distinct_2

Count of unique n-grams divided by total n-grams. Tokenize by whitespace.

```python
tokens = completion.lower().split()
if len(tokens) == 0:
    return 0.0
# For distinct_1 (n=1):
ngrams = tokens
# For distinct_2 (n=2):
# ngrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
score = len(set(ngrams)) / len(ngrams)
```

Parameterize by `n` in `__init__` so one class serves both metrics.

### jensen_shannon_divergence_unigram / bigram

Compare the n-gram distribution of the completion against the reference text.

```python
import numpy as np
from collections import Counter

def _ngram_dist(text, n):
    tokens = text.lower().split()
    if n == 1:
        grams = tokens
    else:
        grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    counts = Counter(grams)
    total = sum(counts.values())
    return {g: c / total for g, c in counts.items()} if total > 0 else {}

def _jsd(dist1, dist2):
    all_keys = list(set(dist1) | set(dist2))
    p = np.array([dist1.get(k, 0.0) for k in all_keys])
    q = np.array([dist2.get(k, 0.0) for k in all_keys])
    m = 0.5 * (p + q)
    # Avoid log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0)
    return float(0.5 * kl_pm.sum() + 0.5 * kl_qm.sum())
```

Parameterize by `n` (1 or 2). Return `0.0` if either text is empty.

### mean_absolute_error

Parse completion and reference as floats (or float lists) and compute MAE.

```python
import numpy as np

try:
    pred = float(completion)
    true = float(reference_text)
    score = abs(pred - true)
except ValueError:
    score = 0.0
```

If the task produces multiple values, parse as comma-separated lists and average the element-wise absolute differences.

### pearson_correlation / spearman_correlation

These are cross-instance by nature (correlation across a dataset), but the HELM pattern computes a
per-instance contribution and then aggregates. For a **per-instance** wrapper, store each instance's
`(pred, reference)` pair as the stat value, then implement `derive_per_instance_stats` to compute
the final correlation.

**Simpler approach (recommended for this benchmark):** Treat each instance as a single `(pred, true)`
pair and report the signed error. Then add a `derive_stats` override that computes the correlation
across all instances using `scipy.stats.pearsonr` or `scipy.stats.spearmanr`.

```python
from scipy.stats import pearsonr, spearmanr

# In evaluate_generation: emit two stats per instance
# Stat(MetricName("pearson_pred")).add(pred_value)
# Stat(MetricName("pearson_true")).add(true_value)

# In derive_per_instance_stats: collect all pred/true pairs, compute correlation
```

Parameterize by correlation type in `__init__`.

### group_match_score

Jaccard similarity between the set of items in the completion and the reference.

```python
pred_set = set(completion.lower().split())
ref_set = set(reference_text.lower().split())
intersection = len(pred_set & ref_set)
union = len(pred_set | ref_set)
score = intersection / union if union > 0 else 0.0
```

If the task uses structured groups (e.g., comma-separated clusters), parse accordingly before computing Jaccard.

### classification_accuracy

Binary: does the completion exactly match the reference label?

```python
score = 1.0 if completion.lower() == reference_text.lower() else 0.0
```

### validity (generic code/structure validity)

Check whether the completion is syntactically valid Python:

```python
import ast
try:
    ast.parse(completion)
    score = 1.0
except SyntaxError:
    score = 0.0
```

For domain-specific validity (NeoCoder), also check constraint satisfaction. If network is available,
fetch reference logic from `https://github.com/JHU-CLSP/NeoCoder/blob/main/src/utils/configs.py#L240`.
Otherwise, implement a basic structural validity check and leave a `# TODO` comment pointing to that URL.

### constraint_satisfaction

Check that the generated code satisfies the stated constraints. If constraints are in annotations:

```python
annotations = request_state.annotations
# Extract constraint results from the annotator
# Default: return fraction of constraints satisfied
```

If constraint logic is unavailable, default to code-validity as a proxy and add a `# TODO`.

### pass_at_1

Binary: does the generated code pass the test cases?

```python
# If test results are in annotations:
annotations = request_state.annotations
score = 1.0 if annotations.get("pass", False) else 0.0
```

If test execution is not available at metric time, treat as annotation-dependent and document this.

### xml_validity

```python
from lxml import etree
try:
    etree.fromstring(completion.encode("utf-8"))
    score = 1.0
except etree.XMLSyntaxError:
    score = 0.0
```

### array_dimensions

Check whether the completion contains a correctly shaped array/matrix. Parse the structure and verify
dimensions match the expected shape from the reference.

```python
import json
try:
    data = json.loads(completion)
    # Verify shape matches expected dimensions from reference
    expected = json.loads(reference_text)
    score = 1.0 if _shape(data) == _shape(expected) else 0.0
except (json.JSONDecodeError, TypeError):
    score = 0.0

def _shape(obj):
    if isinstance(obj, list):
        return (len(obj),) + (_shape(obj[0]) if obj else ())
    return ()
```

### json_validity

```python
import json
try:
    json.loads(completion)
    score = 1.0
except json.JSONDecodeError:
    score = 0.0
```

### creativity_score (CDAT / DSI)

Compute the Divergent Semantic Integration (DSI) score. Reference implementation:
`https://github.com/text-machine-lab/diverse-not-short/blob/main/src/diverse_not_short/util_scripts/eval_utils.py#L327`

Inline fallback — average pairwise cosine distance of word embeddings in the completion:

```python
# Requires a word-embedding model or sentence-transformers
# If unavailable, approximate with type-token ratio as a creativity proxy
tokens = completion.lower().split()
score = len(set(tokens)) / len(tokens) if tokens else 0.0
# TODO: replace with DSI using sentence-transformers when available
```

### iou_score

Intersection-over-Union for bounding boxes. Parse completion and reference as `[x1, y1, x2, y2]`.

```python
import json
try:
    pred = json.loads(completion)
    true = json.loads(reference_text)
    xi1, yi1 = max(pred[0], true[0]), max(pred[1], true[1])
    xi2, yi2 = min(pred[2], true[2]), min(pred[3], true[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_p = (pred[2]-pred[0]) * (pred[3]-pred[1])
    area_t = (true[2]-true[0]) * (true[3]-true[1])
    union = area_p + area_t - inter
    score = inter / union if union > 0 else 0.0
except (json.JSONDecodeError, IndexError, TypeError):
    score = 0.0
```

### layout_quality

Composite heuristic: non-overlapping ratio + alignment score + size-balance.
Reference: `https://github.com/yizhiwang96/TextLogoLayout`. If network is unavailable, implement
a simplified version that checks bounding-box overlap ratios and add a `# TODO`.

### type_token_ratio

```python
tokens = completion.lower().split()
score = len(set(tokens)) / len(tokens) if tokens else 0.0
```

This is a simple whitespace-tokenized TTR. For a spaCy-based version, wrap in a try/except
and fall back to the simple version if spaCy is not installed.

### poem_score, length_score, tone_score, rhyme_score (Vietnamese poem metrics)

Reference implementation: `https://github.com/Anshler/poem_generator/blob/master/utils/check_rule.py#L47`

If network is available, fetch and adapt that code. Otherwise, implement stubs:

- **length_score**: ratio of actual line count to expected line count (from reference).
- **tone_score**: fraction of syllables matching the expected tonal pattern (stub: `0.0` with `# TODO`).
- **rhyme_score**: fraction of line-ending pairs that rhyme (stub: exact-match of last word).
- **poem_score**: weighted average of the above three sub-scores.

These stubs allow the pipeline to run; replace with full Vietnamese phonetic analysis when the reference
code is accessible.

### validity_score (MineAnyBuild)

Critic-model-based validity. Reference: `https://github.com/MineAnyBuild/MineAnyBuild/blob/main/mineanybuild/evaluator.py#L105`

This metric depends on an external critic model. Implement as an **annotation-dependent** metric:

```python
annotations = request_state.annotations
score = float(annotations.get("validity_score", 0.0))
```

Document that the corresponding annotator must be configured separately.

### percentile_rank (WebNovelBench)

> Per the task description, skip PCA-based Composite Score. Focus on LLM-as-a-Judge metrics only.

If the judge score is in annotations:

```python
annotations = request_state.annotations
score = float(annotations.get("percentile_rank", 0.0))
```

If you need to compute percentile rank from raw judge scores, this requires cross-instance data.
Use `derive_per_instance_stats` to collect all scores and compute percentiles:

```python
import numpy as np
# In derive_per_instance_stats:
scores = [stat_value for instance stats]
percentiles = {s: (np.searchsorted(np.sort(scores), s) / len(scores)) * 100 for s in scores}
```

---

## Handling External Dependencies

Some metrics reference external GitHub repositories. Follow this priority:

1. **If network is available:** fetch the reference code, study it, and adapt it into the HELM class.
2. **If network is unavailable:** use the inline implementation provided above. If the inline version
   is a stub or approximation, add a `# TODO: fetch full implementation from <URL>` comment.
3. **Never silently skip a metric.** Always produce a file, even if it contains a stub.

---

## Idempotency

If a metric file already exists in the output directory, **overwrite it** — the skill always regenerates
from the current registry state. This ensures the code stays in sync with any registry changes.

---

## Final Checklist

Before finishing, verify all of the following:

- [ ] Every `null`-class metric in the registry now has a `.py` file in `{root}/metrics/`
- [ ] Every generated file passes `python -c "import ast; ast.parse(...)"`
- [ ] Every generated class inherits from `Metric` (or `EvaluateInstancesMetric` if justified)
- [ ] Every `evaluate_generation` returns `List[Stat]` with the correct metric name string
- [ ] The registry file has been updated: `null` replaced with the actual class name
- [ ] No file references a hardcoded path outside the project root
- [ ] Grouped metrics (distinct_n, jsd, correlation, poem) share a single file with parameterized `__init__`