#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Screener for AI Creativity Benchmarks

Screens papers to identify which ones present AI creativity benchmarks for a systematic
literature review. Uses OpenAI GPT-4.1 via Chat Completions API to evaluate papers against
inclusion criteria covering creativity types, evaluation methods, and K-DOCS/CAQ frameworks.

Features:
  - Parallel async screening with configurable concurrency
  - Checkpoint/resume support for long-running jobs
  - Structured output: JSONL (full) + CSV (spreadsheet for review)
  - Deduplication and idempotent processing
  - Configurable sampling temperature and batch size

Requirements:
  pip install 'openai>=1.50.0,<2.0.0'

Quick Start:
  python paper_screener_gpt41.py

Environment:
  OPENAI_API_KEY: Required

Output Files:
  out/screened_papers.jsonl - Full structured results
  out/screened_papers.csv - Spreadsheet for manual review
  out/screened_papers_checkpoint.jsonl - Checkpoint file (optional, for resume)
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import openai as openai_pkg
    from openai import AsyncOpenAI
    from openai import (
        APIConnectionError,
        APIStatusError,
        RateLimitError,
        BadRequestError,
        AuthenticationError,
    )
except Exception as e:
    print("Error: openai package not installed. Run: pip install 'openai>=1.50.0,<2.0.0'", file=sys.stderr)
    sys.exit(1)

SDK_VERSION = getattr(openai_pkg, "__version__", "unknown")
MODEL_VERSION = "gpt-4.1-2025-04-14"

SCREENING_PROMPT = """You are an expert systematic literature reviewer specializing in AI creativity research.

Your task: Evaluate whether this paper should be INCLUDED in a review of creativity benchmarks for Large Language Models and AI systems.

INCLUSION CRITERIA (paper must meet ALL of these):
1. Focus on AI/LLM Creativity Evaluation
   - Systems: LLMs, foundation models, multimodal models, diffusion models, text-to-image, vision-language models, etc.
   
2. Has a Benchmark/Dataset/Evaluation Component
   - Benchmark, dataset, test suite, evaluation framework, assessment method, leaderboard, shared task
   - Must provide actual evaluation, not just generation without assessment
   
3. Creativity-Related Focus
   - Paper focuses on: divergent thinking, ideation, originality, novelty, creative problem solving, 
     open-ended generation, humor, metaphor, insight, discovery, or related constructs
   - OR evaluates systems on these capabilities
   
4. Open-ended OR Creativity Judgment Tasks
   - Generation tasks: Multiple valid solutions/approaches (stories, ideas, designs, proofs, hypotheses)
   - Judgment tasks: Evaluating creativity of outputs (rating originality, selecting more creative response, 
     assessing novelty/usefulness)
   - Include: Insight-based problem solving even if solutions can be verified
   - Exclude: Pure retrieval, single-solution closed tasks, straightforward QA
   
5. Published 2018–2025
   
6. Empirical Study
   - Reports experimental results with AI systems
   - Not purely theoretical

CREATIVITY TYPES TO INCLUDE:
- Divergent thinking (idea generation, brainstorming, ideation)
- Convergent creativity (insight problems, remote associates, creative connections)
- Creative writing (stories, poetry, narratives, scripts)
- Visual creativity (image generation evaluation, artistic style)
- Humor and joke generation
- Metaphor, analogy, and figurative language generation
- Creative problem solving (open-ended, multiple solutions)
- Scientific creativity (hypothesis generation, experimental design, theory building, discovery)
- Mathematical creativity (novel proof strategies, problem formulation, insight-based approaches)
- Engineering creativity (novel designs, unconventional solutions)
- Creative coding/programming (novel algorithms, creative implementations)
- Multimodal creativity (any combination of text/image/audio/video)
- Advertising and persuasive creativity
- Design thinking and innovation

DOMAIN-SPECIFIC FRAMEWORKS TO RECOGNIZE:
K-DOCS (Kaufman Domains of Creativity Scale) domains:
  - Everyday Creativity
  - Scholarly Creativity
  - Performance Creativity
  - Science Creativity
  - Arts Creativity
  - Other (e.g., technology, business, social innovation)

CAQ (Creative Achievement Questionnaire) domains:
  - Visual Arts
  - Music
  - Dance
  - Architectural Design
  - Creative Writing
  - Scientific Discovery
  - Humor
  - Theater and Film
  - Inventions
  - Culinary Arts
  - Other (e.g., sports, entrepreneurship, games)

EXCLUSION CRITERIA (exclude if ANY apply):
1. Review/Survey papers (unless introducing a new benchmark)
2. Purely closed-ended tasks with single correct answers (factual QA, simple arithmetic)
3. General benchmarks where creativity is minor/incidental subset without specific assessment
4. Non-English papers (but benchmarks testing non-English language creativity are OK)
5. No evaluation component (pure generation without assessment)
6. Not about AI systems (human-only studies)
7. Pure performance benchmarks where creativity is incidental (e.g., code correctness without novelty/originality assessment)

REQUIRED EXTRACTIONS for INCLUDED papers:
- Task: Specific task type, e.g., "story generation", "divergent thinking", "visual creativity evaluation", "mathematical proof generation", "humor generation"
- Metric: How creativity is measured (if available in abstract), e.g., "human ratings of originality", "semantic diversity", "LLM-as-judge", "novelty score", "expert evaluation", "automated fluency/flexibility scoring", or "not listed in abstract"
- K-DOCS Domain: If applicable, specify which K-DOCS domain (Everyday, Scholarly, Performance, Science, Arts, Other)
- CAQ Domain: If applicable, specify which CAQ domain (Visual Arts, Music, Dance, Architectural Design, Creative Writing, Scientific Discovery, Humor, Theater and Film, Inventions, Culinary Arts, Other)

OUTPUT FORMAT (JSON only, no markdown):
{
  "include": true/false,
  "confidence": "high"/"medium"/"low",
  "reasoning": "Brief explanation (1-2 sentences)",
  "task": "task description or null",
  "metric": "creativity measurement or 'not listed in abstract' or null",
  "kdocs_domain": "K-DOCS domain or null",
  "caq_domain": "CAQ domain or null",
  "creativity_type": "divergent/convergent/judgment/generation/mixed or null",
  "flags": ["any concerns or notes"]
}

Be conservative—when uncertain, set confidence to "low" or "medium" for human review.
RETURN ONLY THE JSON OBJECT. NO OTHER TEXT, NO MARKDOWN."""


def load_papers(input_path: Path) -> List[Dict]:
    """Load JSONL file of papers."""
    papers: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return papers


def save_results_jsonl(results: List[Dict], output_path: Path) -> None:
    """Save results as JSONL with sorted keys for reproducibility."""
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def save_summary_csv(results: List[Dict], output_path: Path) -> None:
    """Save results as CSV for easy review."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id", "title", "year", "venue", "include", "confidence",
            "task", "metric", "kdocs_domain", "caq_domain", "creativity_type", "reasoning", "flags",
            "url", "doi", "model_version"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        
        for r in results:
            row = r.copy()
            if "flags" in row and isinstance(row["flags"], list):
                row["flags"] = "; ".join(row["flags"]) if row["flags"] else ""
            writer.writerow(row)


def load_checkpoint(checkpoint_path: Path) -> List[Dict]:
    """Load all records from checkpoint JSONL."""
    results = []
    if not checkpoint_path.exists():
        return results
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                continue
    return results


def _extract_paper_id(paper: Dict) -> str:
    """Consistently extract paper ID from various field names."""
    return (paper.get("s2_paper_id") or 
            paper.get("arxiv_id") or 
            paper.get("paperId") or 
            "")


def print_stats(results: List[Dict]) -> None:
    """Print summary statistics."""
    total = len(results)
    included = sum(1 for r in results if r.get("include") is True)
    excluded = sum(1 for r in results if r.get("include") is False)
    errors = sum(1 for r in results if r.get("error", False))
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total papers:       {total}")
    print(f"Included:           {included} ({100*included/total if total else 0:.1f}%)")
    print(f"Excluded:           {excluded}")
    print(f"Errors:             {errors}")
    
    if included > 0:
        high = sum(1 for r in results if r.get("include") and r.get("confidence") == "high")
        med = sum(1 for r in results if r.get("include") and r.get("confidence") == "medium")
        low = sum(1 for r in results if r.get("include") and r.get("confidence") == "low")
        print(f"\nConfidence (included):")
        print(f"  High:               {high}")
        print(f"  Medium:             {med}")
        print(f"  Low:                {low} (recommend human review)")
    
    if included > 0:
        print(f"\nCreativity types:")
        types = {}
        for r in results:
            if r.get("include"):
                ct = r.get("creativity_type", "unknown")
                types[ct] = types.get(ct, 0) + 1
        for ct, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  {ct}: {count}")
    
    if included > 0:
        print(f"\nK-DOCS Domains:")
        kdocs = {}
        for r in results:
            if r.get("include") and r.get("kdocs_domain"):
                dom = r.get("kdocs_domain")
                kdocs[dom] = kdocs.get(dom, 0) + 1
        for dom, count in sorted(kdocs.items(), key=lambda x: -x[1]):
            print(f"  {dom}: {count}")
    
    if included > 0:
        print(f"\nCAQ Domains:")
        caq = {}
        for r in results:
            if r.get("include") and r.get("caq_domain"):
                dom = r.get("caq_domain")
                caq[dom] = caq.get(dom, 0) + 1
        for dom, count in sorted(caq.items(), key=lambda x: -x[1]):
            print(f"  {dom}: {count}")
    
    print(f"{'='*60}\n")


class PaperScreener:
    """Async screener for AI creativity benchmark papers."""
    
    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 500,
        concurrency: int = 10,
        batch_size: int = 100,
        seed: Optional[int] = None,
        checkpoint_file: Optional[Path] = None,
        progress_every: int = 25,
    ):
        if seed is not None:
            random.seed(seed)
        
        if progress_every <= 0:
            raise ValueError("progress_every must be > 0")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(concurrency)
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.progress_every = progress_every
        self.seed = seed
        
        print(f"[init] model={model}")
        print(f"[init] temperature={temperature}, max_tokens={max_tokens}")
        print(f"[init] concurrency={concurrency}, batch_size={batch_size}")
        print(f"[init] seed={seed if seed is not None else 'None (non-deterministic jitter)'}")
        print(f"[init] model_version={MODEL_VERSION}")
        
        self._seen_ids: Set[str] = set()
        if self.checkpoint_file and self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        pid = rec.get("paper_id")
                        if pid:
                            self._seen_ids.add(pid)
                    except Exception:
                        continue
            if self._seen_ids:
                print(f"[checkpoint] Resuming: skipping {len(self._seen_ids)} already-screened papers")
    
    async def _call_api(self, messages: List[Dict]) -> str:
        """Call OpenAI Chat Completions API."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content if resp and resp.choices else "{}"
    
    async def _screen_one(self, paper: Dict) -> Dict:
        """Screen a single paper."""
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        year = paper.get("year", "Unknown")
        venue = paper.get("venue", "Unknown")
        paper_id = _extract_paper_id(paper)
        
        user_message = f"""Title: {title}
Abstract: {abstract}
Year: {year}
Venue: {venue}"""
        
        messages = [
            {"role": "system", "content": SCREENING_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        tries, delay = 0, 1.0
        first_error_logged = False
        
        while True:
            try:
                async with self.semaphore:
                    text = await self._call_api(messages)
                break
            except (RateLimitError, APIStatusError, APIConnectionError, TimeoutError,
                    BadRequestError, AuthenticationError) as e:
                tries += 1
                if not first_error_logged:
                    print(f"[warn] API error on '{title[:50]}...': {type(e).__name__}", flush=True)
                    first_error_logged = True
                if tries >= 5:
                    return {
                        "paper_id": paper_id, "title": title, "year": year, "venue": venue,
                        "include": None, "confidence": "error",
                        "reasoning": f"API failed after {tries} retries",
                        "error": True
                    }
                await asyncio.sleep(delay + random.random() * 0.25)
                delay = min(delay * 2, 16.0)
            except Exception as e:
                if not first_error_logged:
                    print(f"[warn] Error on '{title[:50]}...': {type(e).__name__}", flush=True)
                    first_error_logged = True
                return {
                    "paper_id": paper_id, "title": title, "year": year, "venue": venue,
                    "include": None, "confidence": "error",
                    "reasoning": str(e)[:200],
                    "error": True
                }
        
        try:
            result = json.loads(text)
        except Exception as e:
            return {
                "paper_id": paper_id, "title": title, "year": year, "venue": venue,
                "include": None, "confidence": "error",
                "reasoning": f"JSON parse error",
                "error": True
            }
        
        result.update({
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "venue": venue,
            "url": paper.get("url"),
            "doi": paper.get("doi"),
            "model_version": MODEL_VERSION,
        })
        
        if "include" not in result:
            result["include"] = None
            result.setdefault("confidence", "low")
            result.setdefault("flags", []).append("missing_include_key")
        
        return result
    
    async def screen_batch(self, papers: List[Dict]) -> List[Dict]:
        """Screen a batch of papers with progress tracking."""
        results: List[Dict] = []
        
        def checkpoint_append(rec: Dict) -> None:
            if not self.checkpoint_file:
                return
            with open(self.checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        
        total = len(papers)
        started = time.time()
        
        async def process_one(idx: int, paper: Dict):
            pid = _extract_paper_id(paper)
            if pid and pid in self._seen_ids:
                return
            
            rec = await self._screen_one(paper)
            results.append(rec)
            checkpoint_append(rec)
            
            processed = idx + 1
            if processed % self.progress_every == 0:
                inc = sum(1 for r in results if r.get("include") is True)
                exc = sum(1 for r in results if r.get("include") is False)
                err = sum(1 for r in results if r.get("error"))
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[{processed}/{total}] {rate:.1f} p/s | incl: {inc} | excl: {exc} | err: {err}")
        
        print(f"\n[start] Screening {total} papers\n")
        
        for start in range(0, total, self.batch_size):
            chunk = papers[start:start + self.batch_size]
            tasks = [process_one(start + i, p) for i, p in enumerate(chunk)]
            await asyncio.gather(*tasks)
        
        elapsed = time.time() - started
        print(f"\n[done] Completed in {elapsed:.1f}s\n")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Screen AI creativity benchmark papers with GPT-4"
    )
    parser.add_argument("--input", type=Path, default=Path("out/harvest.jsonl"), help="Input JSONL file")
    parser.add_argument("--output", type=Path, default=Path("out/screened_papers.jsonl"), help="Output JSONL file")
    parser.add_argument("--model", default="gpt-4.1", help="Model name (default: gpt-4.1)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens per response")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel requests")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=99, help="Random seed for reproducibility")
    parser.add_argument("--progress-every", type=int, default=25, help="Progress update interval")
    parser.add_argument("--checkpoint-file", type=Path, default=None, help="Checkpoint file for resuming")
    
    args = parser.parse_args()
    
    # Load papers
    print(f"[load] Reading {args.input}")
    papers = load_papers(args.input)
    print(f"[load] Found {len(papers)} papers\n")
    
    # Setup checkpoint
    checkpoint = args.checkpoint_file or args.output.with_stem(f"{args.output.stem}_checkpoint")
    if not checkpoint.exists():
        checkpoint.touch()
        print(f"[init] Created checkpoint: {checkpoint}\n")
    else:
        print(f"[init] Using checkpoint: {checkpoint}\n")
    
    # Screen papers
    screener = PaperScreener(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_file=checkpoint,
        progress_every=args.progress_every,
    )
    
    results = asyncio.run(screener.screen_batch(papers))
    
    # Merge checkpoint + results, maintain order, deduplicate
    combined: List[Dict] = []
    seen: Set[str] = set()
    
    for checkpoint_rec in load_checkpoint(checkpoint):
        pid = checkpoint_rec.get("paper_id")
        if pid and pid not in seen:
            combined.append(checkpoint_rec)
            seen.add(pid)
    
    for rec in results:
        pid = rec.get("paper_id")
        if pid and pid not in seen:
            combined.append(rec)
            seen.add(pid)
    
    combined = sorted(combined, key=lambda r: r.get("paper_id", ""))
    
    # Save
    print(f"[save] JSONL: {args.output}")
    save_results_jsonl(combined, args.output)
    
    csv_path = args.output.with_suffix('.csv')
    print(f"[save] CSV:   {csv_path}")
    save_summary_csv(combined, csv_path)
    
    print_stats(combined)


if __name__ == "__main__":
    main()