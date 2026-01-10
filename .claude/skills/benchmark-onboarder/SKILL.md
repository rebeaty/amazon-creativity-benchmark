---
name: benchmark-onboarder
description: Onboard AI creativity benchmarks from academic papers into standardized task implementations. Use when the user wants to onboard a benchmark, create a task.py for a benchmark, process a paper into a benchmark task, or mentions benchmark onboarding.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch
---

# Benchmark Onboarder

Transform academic paper descriptions of AI benchmarks into fully functional, standardized Python task implementations.

## Overview

This skill implements an 8-phase agentic workflow to onboard benchmarks:

1. **Paper Metadata Extraction** - Extract task description, dataset names from PDF
2. **Source Discovery & Validation** - Find and validate HuggingFace/GitHub sources
3. **Repository Context Ingestion** - Extract code, README, evaluation scripts
4. **Dataset Loading** - Load and semantically validate data samples
5. **Evaluation Criteria Extraction** - Determine metrics and evaluation approach
6. **Task Configuration Extraction** - Extract fields, prompts, task type
7. **Code Generation** - Generate task.py with load_data, format_prompt, evaluate
8. **Validation & Pilot Testing** - Test the generated code works correctly

## Required Inputs

Before starting, gather:
- **Benchmark name**: The name of the benchmark (e.g., "BRAINTEASER", "CreativeQA")
- **Paper PDF path**: Path to the academic paper describing the benchmark
- **Output directory**: Where to generate the task files (default: `abc_tasks/`)
- **Optional URLs**: Any known HuggingFace or GitHub URLs for the benchmark

## Benchmark Registry Linkage

**Critical**: Always link benchmarks to their registry entries.

### Primary Sources

1. **batch_all.json** (`ai/harness/batch_all.json`): Master list of benchmarks to onboard
   ```json
   {
     "name": "BRAINTEASER",
     "paper_id": "00900ed6f920fe788fcda25d4d81f5917c0710cd",
     "urls": ["https://github.com/..."]
   }
   ```

2. **benchmark_catalog.json** (`ai/review/benchmark_catalog/benchmark_catalog.json`): Rich metadata
   ```json
   {
     "benchmark_id": "paper_id:BENCHMARK_NAME",
     "benchmark_name": "BRAINTEASER",
     "paper_id": "00900ed6f920fe788fcda25d4d81f5917c0710cd",
     "paper_title": "Full Paper Title",
     "paper_year": "2024",
     "paper_doi": "10.48550/arXiv.xxxx.xxxxx",
     "primary_domain": "Lateral Thinking & Creative Problem Solving",
     "tier": 1
   }
   ```

### Lookup Process

Before onboarding, look up the benchmark in these registries:
1. Search `batch_all.json` by name to get `paper_id` and `urls`
2. Search `benchmark_catalog.json` for richer metadata (domain, tier, DOI)
3. Use the `paper_id` to find the PDF in `ai/review/pdf_cache/{paper_id}.pdf`

### ID Linkage Requirements

The generated `manifest.json` MUST include:
- `paper_id`: The Semantic Scholar paper ID (from registry)
- `benchmark_id`: Composite `{paper_id}:{benchmark_name}`
- `paper_title`: From catalog or extracted from PDF
- `paper_doi`: If available in catalog

## Phase 1: Paper Metadata Extraction

Extract key information from the paper PDF:

```
Read the paper and extract:
- arxiv_id (if available)
- paper_title
- dataset_names (mentioned dataset identifiers)
- task_description (what the benchmark measures)
- expected_data_format (JSON, JSONL, CSV, etc.)
```

## Phase 2: Source Discovery & Validation

Find official data sources:

1. **Check provided URLs first** (most authoritative)
2. **Search HuggingFace** for datasets matching benchmark name
3. **Web search** for official GitHub repositories
4. **Validate each source**:
   - Does the dataset name match the benchmark?
   - Is it from the paper authors?
   - Does the domain match the paper description?

**Critical**: Reject sources that don't match semantically. Example: "D-RAP" (rap lyrics) vs "Draper" (vulnerability detection) - same search term, different benchmarks.

## Phase 3: Repository Context Ingestion

From the GitHub repository, extract:

1. README content
2. Evaluation scripts (look for `eval*.py`, `evaluate*.py`)
3. Data loading code
4. **Sacred prompt**: The original prompt template used in the paper's evaluation

Look for prompt templates in:
- `prompts/`, `templates/` directories
- Variables named `PROMPT`, `TEMPLATE`, `INSTRUCTION`
- Docstrings describing the evaluation format

## Phase 4: Dataset Loading (Agentic)

This phase requires intelligent fallback:

```
Try loading in this order:
1. HuggingFace datasets library (if HF ID found)
2. HuggingFace Hub raw file download
3. GitHub raw URLs for data files
4. Git LFS clone (for large files)
5. Smart fallback: Web search for alternative sources
```

**Semantic Validation**: After loading, verify the data matches the paper:
- Check field names match expected structure
- Verify sample content matches the task domain
- If mismatch detected, try the next data file

**Ground Truth Check**: Detect if labels are hidden (e.g., `"answer": "?"`)

## Phase 5: Evaluation Criteria Extraction

Determine how to evaluate predictions:

| Evaluation Type | Description |
|-----------------|-------------|
| `automatic` | Exact match, F1, accuracy - computable without LLM |
| `llm-judge` | Requires LLM to judge quality (creativity, coherence) |
| `human` | Requires human evaluation |
| `hybrid` | Combination of automatic metrics + LLM judge |

Extract:
- Primary metric (accuracy, F1, BLEU, etc.)
- Secondary metrics
- LLM judge prompt (if applicable)

## Phase 6: Task Configuration Extraction

From repo code and data sample, extract:

```python
TaskSpec:
  class_name: str          # PascalCase class name
  input_field: str         # Field(s) containing input data
  label_field: str         # Field containing ground truth
  prompt_template: str     # Template for formatting prompts
  task_type: str           # classification|generation|ranking|open-ended
  has_media: bool          # True if images/audio/video
  media_type: str          # image|audio|video|mixed
  media_field: str         # Field containing media data
```

## Phase 7: Code Generation

Generate `task.py` following this structure:

```python
"""
{benchmark_name} Benchmark Task
Auto-generated by ABC Onboarder
"""
from shared.abc_base import ABCTask


class {ClassName}(ABCTask):
    """
    {task_description}

    Task type: {task_type}
    Input field: {input_field}
    Label field: {label_field}
    """

    PROMPT_TEMPLATE = """{prompt_template}"""

    def __init__(self):
        super().__init__(
            name="{benchmark_name}",
            task_type="{task_type}",
            input_field="{input_field}",
            label_field="{label_field}",
        )

    def load_data(self, split: str = "test"):
        """Load benchmark data from source."""
        # Implementation varies by data source
        pass

    def format_prompt(self, instance: dict):
        """Format prompt for a single instance.

        Returns:
            str for text-only tasks
            List[dict] for multimodal tasks with content blocks
        """
        pass

    def evaluate(self, prediction: str, reference: str) -> dict:
        """Evaluate prediction against reference.

        Returns:
            dict with 'accuracy' key (0.0-1.0) and optional metrics
        """
        pass
```

### Code Generation Rules

**load_data()**:
- Must fetch from EXTERNAL source (URL, HuggingFace, API)
- Never hardcode sample data
- Return list of dict instances

**format_prompt()**:
- For text-only: return formatted string
- For multimodal: return list of content blocks
  ```python
  [{"type": "text", "text": "..."}, {"type": "image", "url": "..."}]
  ```

**evaluate()**:
- MUST return dict with `"accuracy"` key
- Handle hidden references: `if reference in ('?', '', 'hidden'): return {"accuracy": 0.0, "note": "hidden"}`
- For classification: extract answer with regex from verbose responses
- For generation: use lenient overlap scoring + `{"needs_judge": True}`

## Phase 8: Validation & Pilot Testing

### Execution Pilot
Run the generated code in a subprocess:
1. Import the task class
2. Call `load_data()` - verify returns list with 3+ instances
3. Call `format_prompt()` on samples - verify valid output
4. Call `evaluate()` - verify returns dict

### Live Pilot (if API available)
Run actual LLM inference on 3 instances:
1. Format prompts
2. Get model predictions
3. Evaluate predictions
4. If all scores are 0, run diagnosis

### Diagnosis & Fix Loop
When evaluation fails:
1. Analyze prediction vs reference pairs
2. Identify root cause:
   - `format_mismatch`: Model outputs "Option 1" but reference is "1"
   - `label_field_mismatch`: Wrong field used for ground truth
   - `held_out_test`: References are placeholders
   - `rating_exact_match`: Using exact match for rating tasks
3. Generate fix and re-run pilot

## Output Structure

```
{output_dir}/
├── benchmarks/
│   ├── __init__.py
│   └── {benchmark_name}/
│       ├── __init__.py
│       ├── task.py          # Generated task implementation
│       └── manifest.json    # Metadata and configuration
├── shared/
│   ├── __init__.py
│   └── abc_base.py          # Base class (ABCTask)
└── logs/
    ├── {benchmark}_agent_log.json  # Detailed execution log
    └── _batch_results.json         # Batch run summary
```

## Key Patterns

### Multi-Source Fallback Chain
Always try multiple sources before failing:
1. Provided URLs (authoritative)
2. HuggingFace search
3. Web search for GitHub
4. Git LFS clone
5. Smart LLM-powered discovery

### Semantic Validation Loop
After loading data, always verify:
1. Domain matches paper description
2. Fields exist as expected
3. Sample content is meaningful (not placeholders)

### Error-Driven Fixing
When code fails:
1. Classify the error type
2. Apply targeted fix (not full rewrite)
3. Retry with exponential backoff for rate limits

## Examples

### Example 1: Simple Classification Benchmark

```
User: Onboard the BRAINTEASER benchmark from papers/brainteaser.pdf

Steps:
1. Read paper → Extract: lateral thinking puzzles, multiple choice
2. Search HuggingFace → Find: "baber/brainteaser"
3. Load dataset → Sample: {"question": "...", "answer": "B", "choices": [...]}
4. Generate task.py with classification evaluation
5. Pilot test → Verify works
```

### Example 2: Multimodal Benchmark

```
User: Onboard the MM-Vet benchmark

Steps:
1. Read paper → Extract: vision-language evaluation, image understanding
2. Find source → GitHub with image URLs
3. Detect multimodal → has_media=True, media_type="image"
4. Generate format_prompt returning content blocks
5. Pilot with multimodal model
```

### Example 3: Generation Benchmark with LLM Judge

```
User: Onboard CreativeWriting benchmark

Steps:
1. Read paper → Extract: creative story generation, no ground truth
2. Detect evaluation_type="llm-judge"
3. Extract judge prompt from paper
4. Generate evaluate() that returns {"accuracy": 0.0, "needs_judge": True}
5. Store judge_prompt in manifest.json
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No sources found | Benchmark too new/obscure | Try web search with paper title + "dataset" |
| Semantic mismatch | Wrong dataset with similar name | Try next data file, validate domain |
| All scores 0 | Format mismatch or hidden labels | Run diagnosis, check reference values |
| Import error | Missing package | Install required package and retry |
| Timeout | Loading too much data | Add limit to load_data |

## Best Practices

1. **Always read the paper first** - Understand what the benchmark measures
2. **Validate semantically** - Don't trust dataset names alone
3. **Use the sacred prompt** - Original prompts from papers are most reliable
4. **Handle multimodal properly** - Return content blocks, not concatenated strings
5. **Include needs_judge flag** - For subjective tasks requiring LLM evaluation
6. **Log everything** - Save decisions and reasoning for debugging
7. **Test with actual data** - Pilot testing catches issues early
