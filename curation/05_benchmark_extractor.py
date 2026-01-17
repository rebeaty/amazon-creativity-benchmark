#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Extraction

Extracts creativity benchmarks from PDFs with detailed task metadata.

EXPECTED FILE LOCATIONS (in script root directory):
- merged_prisma_papers.csv (input paper metadata)
- pdf_cache/ (directory containing PDFs named by paper_id.pdf)
- pi_decisions.json (PI override decisions for 16 uncertain papers)
"""

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf

try:
    import google.genai as genai
    from google.genai.types import GenerateContentConfig
except ImportError:
    raise ImportError("Install google-genai: pip install google-genai")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_TEMPERATURE = 0.3
DEFAULT_CONCURRENCY = 5

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

SYSTEM_INSTRUCTION = """You are extracting creativity benchmarks from AI/LLM research papers.

Your task is to identify all benchmarks and tasks that THIS PAPER ACTUALLY USES in its experiments/evaluation.

═══════════════════════════════════════════════════════════════════════════

STEP 1: FIND WHAT WAS USED

Read the Methods, Evaluation, Experiments, and Results sections.

INCLUDE benchmarks/tasks that are:
✓ Actually tested/evaluated in this paper's experiments
✓ Used to generate results presented in this paper

EXCLUDE benchmarks/tasks that are:
✗ Only mentioned in background/related work
✗ Cited as prior work but not tested
✗ Discussed but not actually used

═══════════════════════════════════════════════════════════════════════════

STEP 2: ASSESS RELEVANCE

For each benchmark found, determine its relevance to AI creativity evaluation:

"relevant" = ALL of these are true:
  • Presents a benchmark/dataset/evaluation for AI/LLM systems
  • Focuses on creativity (divergent thinking, ideation, novelty, humor, etc.)
  • Uses open-ended tasks (multiple solutions) OR creativity judgment tasks
  • Reports empirical results with AI systems

"potentially_irrelevant" = ANY of these are true:
  • Only theoretical discussion without evaluation framework
  • Purely closed-ended tasks (single correct answer, retrieval)
  • No creativity focus—this is a CREATIVITY benchmark catalog, NOT general AI capabilities
  • Survey/review without novel benchmark

WELL-KNOWN GENERAL BENCHMARKS (always mark "potentially_irrelevant"):
  These test general AI capabilities, NOT creativity:
  • Reasoning: ARC (Abstraction and Reasoning Corpus), 1D-ARC, BIG-Bench, GSM8K, MATH
  • Knowledge: MMLU, TriviaQA, NaturalQuestions, SQuAD
  • Commonsense: HellaSwag, WinoGrande, CommonsenseQA, PIQA, SIQA, COPA
  • Code: HumanEval, MBPP, APPS, SyGuS, CodeContests
  • Vision: ImageNet, COCO (object detection), VQA, GQA, CLEVR
  • Language: LAMBADA, BoolQ, RTE, WIC

  Even if these appear in a creativity paper, they are NOT creativity benchmarks.

"uncertain" = Creativity focus is marginal or unclear

IMPORTANT: Include ALL benchmarks used in the paper, even if "potentially_irrelevant". 
We filter later—your job is to extract accurately.

═══════════════════════════════════════════════════════════════════════════

STEP 3: DETERMINE STATUS

For each benchmark and task:

"new" = This paper introduces it for the first time
"reused" = Uses existing benchmark/task from prior work (unmodified)
"extended" = Modifies/adapts existing benchmark/task from prior work
"unclear" = Cannot determine from available information

═══════════════════════════════════════════════════════════════════════════

STEP 4: CHECK FOR FORMAL NAMES

has_formal_name:
  true = Has a proper named convention (e.g., "TaleBrush", "Torrance Test")
  false = Only descriptive (e.g., "story generation task", "divergent thinking test")

═══════════════════════════════════════════════════════════════════════════

STEP 5: CLASSIFY EACH TASK

For each task, extract:

1. TASK TYPE:
   - "generate": AI generates creative content
   - "evaluate": AI judges/rates creativity
   - "mixed": Both generation and evaluation
   - "other": Different type (specify in notes)

2. MODEL TYPE (what AI was tested):
   - "LLM": Large language models (text-based)
   - "VLM": Vision-language models (multimodal)
   - "Diffusion": Image/video generation models
   - "Audio": Audio/music generation models
   - "Multimodal": Multiple model types tested
   - "Other": Specify type

3. MODALITY:
   - "text" | "image" | "audio" | "video" | "multimodal" | "other"

4. SCORING METHOD:
   - "human": Human raters/judges (including crowdworkers)
   - "llm": LLM judges or AI-based evaluation
   - "automatic": Automated NLP/CV metrics (BLEU, ROUGE, F1, accuracy, perplexity, cosine similarity, etc.)
   - "mixed": Combination of methods (e.g., human + automatic, or human + LLM)
   - "none": Dataset only; no model evaluation run yet
   - "unknown": Paper does not specify

   IMPORTANT: "automatic" means NLP/CV metrics, NOT LLM judges. LLM-as-judge is "llm".

5. SCORING DETAIL:
   Be SPECIFIC and CONCISE. State only what is measured.
   Examples: "BLEU score", "Originality rating 1-7", "Cosine similarity", "GPT-4 judge", "Humor rating"

═══════════════════════════════════════════════════════════════════════════

OUTPUT FORMAT

Return ONLY valid JSON (no markdown, no code blocks):

{{
  "benchmarks": [
    {{
      "name": "benchmark name",
      "has_formal_name": true | false,
      "status": "new | reused | extended | unclear",
      "relevance": "relevant | potentially_irrelevant | uncertain",
      "relevance_reason": "Brief explanation",
      "origin_name": "original name if reused/extended, else null",
      "origin_paper": "brief citation if reused/extended, else null",
      "tasks": [
        {{
          "name": "task name",
          "has_formal_name": true | false,
          "description": "brief description (1-2 sentences)",
          "status": "new | reused | extended | unclear",
          "origin_name": "original task name if reused/extended, else null",
          "origin_paper": "citation if reused/extended, else null",
          "task_type": "generate | evaluate | mixed | other",
          "task_type_notes": "if other, specify what type",
          "model_type": "LLM | VLM | Diffusion | Audio | Multimodal | Other",
          "model_type_notes": "if Other, specify",
          "modality": "text | image | audio | video | multimodal | other",
          "scoring_method": "human | llm | automatic | mixed | none | unknown",
          "scoring_detail": "specific metric(s) used"
        }}
      ]
    }}
  ]
}}

If no benchmarks were actually used (only discussed), return: {{"benchmarks": []}}"""

USER_PROMPT_TEMPLATE = """Extract all benchmarks from this paper:

PAPER METADATA:
Title: {title}
Year: {year}
DOI: {doi}

PAPER TEXT:
{paper_text}"""

# ──────────────────────────────────────────────────────────────────────────────
# PDF & JSON HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def pdf_to_text(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF."""
    try:
        doc = pymupdf.open(str(pdf_path))
        try:
            text = "\n\n".join(page.get_text("text") for page in doc)
            return text
        finally:
            doc.close()
    except Exception as e:
        print(f"  ⚠ PDF conversion failed: {e}")
        return None


def parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse Gemini JSON response."""
    if not raw:
        return None
    
    text = raw.strip()
    
    # Strip markdown
    if "```" in text:
        if "```json" in text:
            text = text.split("```json", 1)[1]
        else:
            text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()
    
    # Find JSON
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1:
        try:
            return json.loads(text[first:last + 1])
        except Exception:
            pass
    
    return None


# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkExtractor:
    def __init__(self, api_key: str, model: str, concurrency: int):
        if not api_key:
            raise ValueError("GEMINI_API_KEY required")
        
        self.client = genai.Client(api_key=api_key).aio
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
    
    async def call_gemini(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Call Gemini with system instruction and user prompt."""
        try:
            async with self.semaphore:
                response = await self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=GenerateContentConfig(
                        temperature=GEMINI_TEMPERATURE,
                        system_instruction=SYSTEM_INSTRUCTION
                    )
                )
            
            result = parse_json_response(response.text if response else "")
            
            # Retry once if parse failed
            if result is None:
                retry_prompt = user_prompt + "\n\nIMPORTANT: Reply with ONLY valid JSON, no markdown."
                async with self.semaphore:
                    response = await self.client.models.generate_content(
                        model=self.model,
                        contents=retry_prompt,
                        config=GenerateContentConfig(
                            temperature=GEMINI_TEMPERATURE,
                            system_instruction=SYSTEM_INSTRUCTION
                        )
                    )
                result = parse_json_response(response.text if response else "")
            
            return result
        except Exception as e:
            print(f"  ⚠ Gemini error: {type(e).__name__}")
            print(f"     Details: {str(e)}")
            return None
    
    async def extract_benchmarks(self, paper_text: str, title: str, year: str, doi: str) -> List[Dict[str, Any]]:
        """Extract benchmarks from paper text."""
        # Truncate if too long
        MAX_CHARS = 400_000
        if len(paper_text) > MAX_CHARS:
            paper_text = paper_text[:MAX_CHARS] + "\n\n[TRUNCATED]"
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=title or "Unknown",
            year=year or "Unknown",
            doi=doi or "None",
            paper_text=paper_text
        )
        
        result = await self.call_gemini(user_prompt)
        if result and isinstance(result, dict):
            return result.get("benchmarks", [])
        return []


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def load_papers_with_pdfs(csv_path: str, cache_dir: Path, pi_decisions_json: Optional[str] = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load papers that have PDFs available in cache.
    
    Returns:
        (papers_with_pdfs, papers_without_pdfs)
    """
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    # Load PI decisions (overrides for uncertain papers)
    pi_decisions = {}
    if pi_decisions_json and Path(pi_decisions_json).exists():
        print(f"Loading PI decisions from: {pi_decisions_json}")
        with open(pi_decisions_json, encoding="utf-8") as f:
            pi_decisions = json.load(f)
        print(f"  Loaded {len(pi_decisions)} PI override decision(s)")
    
    # Get all PDFs in cache (stem = filename without .pdf)
    pdf_files = {p.stem: p for p in cache_dir.glob("*.pdf")}
    print(f"Found {len(pdf_files)} PDFs in cache: {cache_dir}")
    
    papers_with_pdfs = []
    papers_without_pdfs = []
    
    for row in rows:
        paper_id = (row.get("paper_id") or "").strip()
        
        # Check PI decisions first
        if paper_id in pi_decisions:
            decision = pi_decisions[paper_id].lower()
            if decision != "include":
                continue
        else:
            # Fall back to CSV logic
            final = (row.get("final_decision") or "").strip().lower()
            if final != "include":
                n_include = int(row.get("n_include") or 0)
                n_exclude = int(row.get("n_exclude") or 0)
                if not (n_include > 0 and n_exclude == 0):
                    continue
        
        # Match PDF by paper_id
        if paper_id and paper_id in pdf_files:
            row["_pdf_path"] = pdf_files[paper_id]
            papers_with_pdfs.append(row)
            print(f"  [OK] Matched: {paper_id}.pdf")
        else:
            papers_without_pdfs.append(row)
            if paper_id:
                print(f"  [MISS] Missing: {paper_id}.pdf")
            else:
                print(f"  [MISS] No paper_id for: {row.get('title', 'Unknown')[:50]}")
    
    print(f"\nMatched {len(papers_with_pdfs)} papers with PDFs")
    if papers_without_pdfs:
        print(f"Missing PDFs for {len(papers_without_pdfs)} papers")
    
    return papers_with_pdfs, papers_without_pdfs


async def extract_all_benchmarks(papers: List[Dict[str, Any]], extractor: BenchmarkExtractor, output_dir: Path) -> List[Dict[str, Any]]:
    """Extract benchmarks from all papers concurrently, saving incrementally to JSONL."""
    total = len(papers)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "extracted_benchmarks.jsonl"
    
    # Lock for thread-safe file writing
    write_lock = asyncio.Lock()
    completed = 0
    
    async def process_one(idx: int, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single paper."""
        nonlocal completed
        
        title = paper.get("title", "").strip()
        paper_id = paper.get("paper_id", "")
        
        # Extract text (fast, synchronous)
        pdf_path = paper["_pdf_path"]
        paper_text = pdf_to_text(pdf_path)
        
        if not paper_text:
            record = {
                "paper_id": paper_id,
                "title": title,
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "error": "PDF text extraction failed",
                "benchmarks": []
            }
        else:
            # Extract benchmarks (slow, async - this is where concurrency helps)
            benchmarks = await extractor.extract_benchmarks(
                paper_text,
                title,
                paper.get("year", ""),
                paper.get("doi", "")
            )
            
            record = {
                "paper_id": paper_id,
                "title": title,
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "benchmarks": benchmarks
            }
        
        # Thread-safe progress tracking and file writing
        async with write_lock:
            completed += 1
            print(f"[{completed}/{total}] {title[:60]}... -> {len(record.get('benchmarks', []))} benchmark(s)")
            
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return record
    
    print(f"\nStarting concurrent extraction of {total} papers...")
    print(f"Concurrency limit: {extractor.semaphore._value}\n")
    
    # Create all tasks and run them concurrently
    tasks = [process_one(i, paper) for i, paper in enumerate(papers)]
    results = await asyncio.gather(*tasks)
    
    print(f"\n[OK] All extractions complete. Results saved to: {jsonl_path}")
    
    return results


def save_missing_pdfs_report(papers_without_pdfs: List[Dict[str, Any]], output_dir: Path):
    """Save report of papers with missing PDFs."""
    if not papers_without_pdfs:
        return
    
    missing_path = output_dir / "missing_pdfs.csv"
    
    # Get all column names from the source data
    all_keys = set()
    for row in papers_without_pdfs:
        all_keys.update(row.keys())
    
    # Remove internal keys
    all_keys.discard("_pdf_path")
    
    # Sort keys for consistent output
    columns = sorted(list(all_keys))
    
    with open(missing_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        for row in papers_without_pdfs:
            # Remove internal keys before writing
            clean_row = {k: v for k, v in row.items() if k != "_pdf_path"}
            writer.writerow(clean_row)
    
    print(f"[OK] Saved missing PDFs report to: {missing_path}")


def save_extraction_results(records: List[Dict[str, Any]], output_dir: Path):
    """Save extraction results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save flattened CSV
    csv_path = output_dir / "extracted_benchmarks.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "paper_id", "paper_title", "paper_year", "paper_doi",
            "benchmark_name", "benchmark_has_formal_name", "benchmark_status",
            "benchmark_relevance", "benchmark_relevance_reason",
            "task_name", "task_has_formal_name", "task_description", "task_status",
            "task_type", "task_type_notes",
            "model_type", "model_type_notes",
            "modality", 
            "scoring_method", "scoring_detail"
        ])
        
        for record in records:
            if record.get("error"):
                continue
            
            for bench in record.get("benchmarks", []):
                bench_name = bench.get("name", "")
                bench_status = bench.get("status", "")
                bench_formal = bench.get("has_formal_name", True)
                relevance = bench.get("relevance", "uncertain")
                relevance_reason = bench.get("relevance_reason", "")
                
                for task in bench.get("tasks", []):
                    writer.writerow([
                        record.get("paper_id"),
                        record.get("title"),
                        record.get("year"),
                        record.get("doi"),
                        bench_name,
                        bench_formal,
                        bench_status,
                        relevance,
                        relevance_reason,
                        task.get("name"),
                        task.get("has_formal_name", True),
                        task.get("description"),
                        task.get("status"),
                        task.get("task_type"),
                        task.get("task_type_notes", ""),
                        task.get("model_type"),
                        task.get("model_type_notes", ""),
                        task.get("modality"),
                        task.get("scoring_method"),
                        task.get("scoring_detail")
                    ])
    
    print(f"[OK] Saved extraction results to: {csv_path}")


async def main_async(args):
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    
    print(f"\n> PDF cache: {cache_dir.resolve()}")
    print(f"> Output dir: {output_dir.resolve()}")
    print(f"> Model: {GEMINI_MODEL}")
    print(f"> Concurrency: {args.concurrency}\n")
    
    # Load papers with PDFs
    papers, papers_without_pdfs = load_papers_with_pdfs(args.input_csv, cache_dir, args.pi_decisions)
    if not papers:
        print("No papers with PDFs found!")
        return
    
    # Initialize extractor
    extractor = BenchmarkExtractor(GEMINI_API_KEY, GEMINI_MODEL, args.concurrency)
    
    # Extract benchmarks
    print(f"\n{'='*70}")
    print("EXTRACTING BENCHMARKS FROM PAPERS")
    print(f"{'='*70}")
    
    records = await extract_all_benchmarks(papers, extractor, output_dir)
    
    # Save results
    save_extraction_results(records, output_dir)
    save_missing_pdfs_report(papers_without_pdfs, output_dir)
    
    # Summary
    n_total = sum(len(r.get("benchmarks", [])) for r in records)
    n_tasks = sum(len(task) for r in records for bench in r.get("benchmarks", []) for task in bench.get("tasks", []))
    
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"Papers processed: {len(records)}")
    print(f"Total benchmarks extracted: {n_total}")
    print(f"Total tasks extracted: {n_tasks}")
    print(f"Missing PDFs: {len(papers_without_pdfs)}")
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'extracted_benchmarks.jsonl'}")
    print(f"  - {output_dir / 'extracted_benchmarks.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Extract creativity benchmarks with detailed metadata")
    parser.add_argument("--input-csv", default="merged_prisma_papers.csv", 
                       help="Input CSV with papers (default: merged_prisma_papers.csv)")
    parser.add_argument("--cache-dir", default="pdf_cache", help="PDF cache directory")
    parser.add_argument("--output-dir", default="extraction_output", help="Output directory")
    parser.add_argument("--pi-decisions", default="pi_decisions.json",
                       help="Path to pi_decisions.json with PI override decisions (default: pi_decisions.json)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="API concurrency")
    
    args = parser.parse_args()
    
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()