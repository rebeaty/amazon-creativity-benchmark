# Amazon Creativity Benchmark

A pipeline for curating AI creativity benchmarks from academic literature. This project systematically discovers, screens, and catalogs benchmarks that evaluate creative AI capabilities.

## What This Is

This repository contains tools and data for building a comprehensive catalog of AI creativity benchmarks. The pipeline:

1. **Harvests** thousands of papers from Semantic Scholar using targeted queries
2. **Screens** papers with GPT-4 to identify those presenting actual benchmarks
3. **Verifies** that identified benchmarks have publicly accessible datasets
4. **Downloads** PDFs of included papers for detailed analysis
5. **Extracts** structured metadata about each benchmark

The result is a curated collection of creativity benchmarks with their source papers, ready for further analysis or implementation.

---

## Directory Structure

```
amazon_creativity_benchmark/
├── README.md
├── CLAUDE.md                    # Project-wide Claude context
├── .env.template
│
├── curation/                    # 5-stage paper screening pipeline
│   ├── 01_lit_harvester_s2.py      # Harvest from Semantic Scholar
│   ├── 02_paper_screener_gpt41.py  # Screen benchmark for relevance
│   ├── 03_dataset_verifier.py      # Verify public datasets exist
│   ├── 04_pdf_downloader_gemini.py # Download paper PDFs
│   └── 05_benchmark_extractor.py   # Extract benchmark metadata
│
├── data/
│   └── onboarding_ready/        # Final screened benchmark list
│
└── .claude/skills/benchmark-onboarder/
    ├── SKILL.md                 # Claude Code skill instructions
    ├── LEARNINGS.md             # Team-accumulated benchmark quirks
    ├── TEAM-OVERVIEW.md         # Human team orientation
    ├── helm-template.md         # HELM Scenario code patterns
    ├── benchmarks.json          # Benchmark queue with GA assignments
    └── examples/                # Working scenario examples
        ├── brainteaser.py
        ├── analobench.py
        └── riddlesense.py
```

---

## The Curation Pipeline

The curation pipeline consists of five scripts that run in sequence. Each script is checkpointed, so you can resume if interrupted.

### Step 1: Harvest Papers
```bash
python curation/01_lit_harvester_s2.py
```
Queries Semantic Scholar for papers on AI creativity benchmarks (2018-2025). Uses the Graph API with rate limiting and pagination. Outputs `harvest.jsonl`.

### Step 2: Screen for Benchmarks
```bash
python curation/02_paper_screener_gpt41.py
```
Uses GPT-4 to evaluate each paper: Does it present a benchmark? Is it about creativity/reasoning? Parallel async processing with checkpointing. Outputs `screened_papers.jsonl`.

### Step 3: Verify Dataset Availability
```bash
python curation/03_dataset_verifier.py
```
For each screened paper, uses Gemini with Google Search grounding to verify the benchmark has a publicly accessible dataset. Outputs `verified_papers.jsonl`.

### Step 4: Download PDFs
```bash
python curation/04_pdf_downloader_gemini.py
```
Downloads PDFs for all verified papers. Uses Gemini to find PDF links when direct URLs aren't available. Papers are saved to `data/pdf_cache/` named by their Semantic Scholar ID.

### Step 5: Extract Benchmark Metadata
```bash
python curation/05_benchmark_extractor.py
```
Reads each PDF and extracts structured metadata: task description, dataset format, evaluation metrics, etc. Uses Gemini-2.5-Pro for extraction. Outputs `extracted_benchmarks.jsonl`.

---

## Setup

### API Keys Required

Copy `.env.template` to `.env` and add your keys:

```bash
OPENAI_API_KEY=sk-...      # For paper screening (GPT-4)
GOOGLE_API_KEY=...         # For verification & extraction (Gemini)
HF_TOKEN=hf_...            # Optional: for private HuggingFace datasets
```

### Dependencies

```bash
pip install openai google-generativeai httpx tenacity
pip install datasets  # for HuggingFace integration
```

---

## Benchmark Onboarding

Once benchmarks are curated, the next step is **onboarding** them into HELM Scenario implementations. This is handled by a Claude Code skill documented in:

**[.claude/skills/benchmark-onboarder/](.claude/skills/benchmark-onboarder/)**

The skill follows a 5-step workflow:

1. **Qualify** — Verify it's a creativity benchmark with extractable prompts
2. **Examine dataset** — Identify fields to use vs. skip
3. **Check instructions** — Find task wording from paper/README if specified
4. **Generate scenario** — Produce standardized `scenario.py` following HELM patterns
5. **Verify** — Confirm code runs, fields map correctly

See [TEAM-OVERVIEW.md](.claude/skills/benchmark-onboarder/TEAM-OVERVIEW.md) for team workflow and GA assignments.

---

## License

MIT
