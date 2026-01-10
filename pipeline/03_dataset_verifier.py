#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Availability Verifier for AI Creativity Benchmarks

Verifies if screened papers have publicly accessible datasets/benchmarks
using Gemini-2.5-Flash with Google Search grounding for web access.

Features:
  - Parallel verification with configurable concurrency
  - Checkpoint/resume support for long-running jobs
  - Exponential backoff with seeded random jitter for retry robustness
  - Input-order preservation across async batching
  - Deterministic temperature/seed control for reproducibility
  - Streaming batch output with progress tracking
  - Dual format output (JSONL + CSV) with consistent field ordering

Requirements:
  pip install 'google-genai>=0.4.0,<0.5.0'

Quick Start:
  python dataset_verifier_gemini.py \\
    --input screened_papers.jsonl \\
    --output verified_papers.jsonl \\
    --temperature 1.0 \\
    --seed 99

Environment:
  GOOGLE_API_KEY or GEMINI_API_KEY: Required for Gemini API access

Output Files:
  verified_papers.jsonl - Newline-delimited results with search queries and metadata
  verified_papers.csv - Spreadsheet view of key fields
  verified_papers_checkpoint.jsonl - Checkpoint file (for resume on interruption)

Tuning:
  --concurrency: Parallel verification tasks (default: 10, adjust for rate limits)
  --batch-size: Papers per batch (default: 100, affects memory/API pacing)
  --temperature: Gemini response synthesis (1.0=Google recommended for grounding, default: 1.0)
  --seed: Random seed for retry jitter consistency (default: 99)
  --progress-every: Print progress every N papers (default: 25)
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
    from google import genai
    from google.genai.types import (
        GenerateContentConfig,
        GoogleSearch,
        Tool,
    )
except ImportError:
    print("Error: google-genai not installed. Run: pip install 'google-genai>=0.4.0,<0.5.0'", file=sys.stderr)
    sys.exit(1)

SDK_VERSION = getattr(genai, "__version__", "unknown")
MODEL_VERSION = "gemini-2.5-flash"

VERIFICATION_PROMPT = """You are verifying if an AI creativity benchmark paper has publicly accessible resources.

PAPER:
Title: {title}
Year: {year}

GOAL: Find if this paper has an accessible dataset/benchmark/evaluation resource.

SEARCH STRATEGY:
1. Search: "{title} dataset"
2. Search: "{title} github"
3. Check: Papers with Code, HuggingFace, GitHub, project pages

IMPORTANT:
- Focus on finding the primary download/access link
- Verify it's for THIS specific paper (check title/authors match)
- Prefer canonical sources: HuggingFace > GitHub > Papers with Code > project pages
- If multiple links, prioritize the most direct/persistent one
- If dataset is available, read the README/description to understand what's in it

RETURN ONLY THIS JSON (no markdown, no explanation, just the JSON object):
{{
  "dataset_available": true/false,
  "primary_url": "direct link to dataset/repo or null",
  "access_method": "huggingface/github/download/papers_with_code/project_page/none",
  "dataset_description": "If available: 1-2 sentence summary of dataset contents. Include: number of items/tasks, what type (prompts, images, examples), evaluation method, human vs automated ratings. If unavailable: null",
  "has_code": true/false,
  "access_barriers": ["e.g., registration, broken_link, paywall"],
  "notes": "Brief explanation - why available/unavailable, any special requirements",
  "confidence": "high/medium/low",
  "needs_manual_review": true/false
}}

CRITICAL: Return ONLY the JSON object above. No other text, no markdown formatting, no explanation.

If you can't find a dataset, set dataset_available=false and explain why in notes."""


def _extract_paper_id(paper: Dict) -> str:
    """Consistently extract paper ID from various field names."""
    return (paper.get("paper_id") or 
            paper.get("s2_paper_id") or 
            paper.get("arxiv_id") or 
            paper.get("paperId") or 
            "")


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


def save_results_jsonl(results: List[Dict], output_path: Path) -> None:
    """Save results as JSONL with sorted keys for reproducibility."""
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def save_summary_csv(results: List[Dict], output_path: Path) -> None:
    """Save results as CSV for easy review."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "paper_id", "title", "year", "venue", "include",
            "dataset_available", "primary_url", "access_method",
            "dataset_description", "has_code", "confidence",
            "needs_manual_review", "notes", "access_barriers_csv",
            "verified_at", "model_version"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        
        for r in results:
            if r.get("verification_skipped"):
                continue
            row = r.copy()
            if "access_barriers" in row and isinstance(row["access_barriers"], list):
                row["access_barriers_csv"] = "; ".join(row["access_barriers"]) if row["access_barriers"] else ""
            writer.writerow(row)


def print_stats(results: List[Dict]) -> None:
    """Print summary statistics."""
    verified = [r for r in results if r.get("include") and not r.get("verification_skipped")]
    
    available = sum(1 for r in verified if r.get("dataset_available") is True)
    unavailable = sum(1 for r in verified if r.get("dataset_available") is False)
    errors = sum(1 for r in verified if r.get("verification_error"))
    manual = sum(1 for r in verified if r.get("needs_manual_review"))
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total papers:           {len(results)}")
    print(f"Verified:               {len(verified)}")
    print(f"  Dataset available:    {available} ({100*available/len(verified) if verified else 0:.1f}%)")
    print(f"  Not available:        {unavailable}")
    print(f"  Verification errors:  {errors}")
    print(f"  Need manual review:   {manual}")
    
    if available:
        high = sum(1 for r in verified if r.get("dataset_available") and r.get("confidence") == "high")
        med = sum(1 for r in verified if r.get("dataset_available") and r.get("confidence") == "medium")
        low = sum(1 for r in verified if r.get("dataset_available") and r.get("confidence") == "low")
        print(f"\nConfidence (available):")
        print(f"  High:                 {high}")
        print(f"  Medium:               {med}")
        print(f"  Low:                  {low}")
    
    if available:
        methods = {}
        for r in verified:
            if r.get("dataset_available"):
                m = r.get("access_method", "unknown")
                methods[m] = methods.get(m, 0) + 1
        
        print(f"\nAccess methods:")
        for method, count in sorted(methods.items(), key=lambda x: -x[1]):
            print(f"  {method}: {count}")
    
    print(f"{'='*60}\n")


class GeminiDatasetVerifier:
    """Async verifier for dataset availability in AI creativity papers."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        concurrency: int = 10,
        batch_size: int = 100,
        seed: int = 99,
        checkpoint_file: Optional[Path] = None,
        progress_every: int = 25,
    ):
        if seed is not None:
            random.seed(seed)
        
        if progress_every <= 0:
            raise ValueError("progress_every must be > 0")
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(concurrency)
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.progress_every = progress_every
        self.seed = seed
        self.search_tool = Tool(google_search=GoogleSearch())
        
        print(f"[init] model={model}")
        print(f"[init] temperature={temperature}, concurrency={concurrency}")
        print(f"[init] batch_size={batch_size}, seed={seed}")
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
                print(f"[checkpoint] Resuming: skipping {len(self._seen_ids)} already-verified papers")
    
    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with exponential backoff."""
        delay = 1.0
        for attempt in range(5):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=GenerateContentConfig(
                        tools=[self.search_tool],
                        temperature=self.temperature,
                    ),
                )
                return response.text if response else ""
            except Exception as e:
                if attempt == 4:
                    raise
                wait_time = delay + random.random() * 0.25
                await asyncio.sleep(wait_time)
                delay = min(delay * 2, 16.0)
    
    async def _verify_one(self, paper: Dict) -> Dict:
        """Verify a single paper."""
        title = paper.get("title", "") or ""
        paper_id = _extract_paper_id(paper)
        year = paper.get("year", "")
        
        doi = paper.get("doi") or ""
        pub_url = paper.get("url") or ""
        
        prompt = VERIFICATION_PROMPT.format(title=title, year=year)
        if doi or pub_url:
            prompt += f"\n\nAdditional context:\nDOI: {doi}\nPublisher URL: {pub_url}"
        
        tries, delay = 0, 1.0
        first_error_logged = False
        
        while True:
            try:
                async with self.semaphore:
                    result_text = await self._call_gemini(prompt)
                break
            except Exception as e:
                tries += 1
                if not first_error_logged:
                    print(f"[warn] Gemini error on '{title[:50]}...': {type(e).__name__}", flush=True)
                    first_error_logged = True
                if tries >= 5:
                    return {
                        "paper_id": paper_id, "title": title, "year": year,
                        "include": paper.get("include"),
                        "dataset_available": None, "confidence": "error",
                        "notes": f"API failed after {tries} retries",
                        "verification_error": True,
                        "model_version": MODEL_VERSION,
                    }
                await asyncio.sleep(delay + random.random() * 0.25)
                delay = min(delay * 2, 16.0)
        
        try:
            # Strip markdown if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
        except Exception as e:
            return {
                "paper_id": paper_id, "title": title, "year": year,
                "include": paper.get("include"),
                "dataset_available": None, "confidence": "error",
                "notes": "JSON parse failed",
                "verification_error": True,
                "model_version": MODEL_VERSION,
            }
        
        result.update({
            "paper_id": paper_id,
            "title": title,
            "year": year,
            "include": paper.get("include"),
            "venue": paper.get("venue"),
            "verified_at": time.strftime("%Y-%m-%d"),
            "model_version": MODEL_VERSION,
            "verification_error": False,
        })
        
        return result
    
    async def verify_batch(self, papers: List[Dict]) -> List[Dict]:
        """Verify papers in batches with progress tracking."""
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
            
            rec = await self._verify_one(paper)
            results.append(rec)
            checkpoint_append(rec)
            
            processed = idx + 1
            if processed % self.progress_every == 0:
                avail = sum(1 for r in results if r.get("dataset_available") is True)
                unavail = sum(1 for r in results if r.get("dataset_available") is False)
                err = sum(1 for r in results if r.get("verification_error"))
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[{processed}/{total}] {rate:.1f} p/s | avail: {avail} | unavail: {unavail} | err: {err}")
        
        print(f"\n[start] Verifying {total} papers\n")
        
        for start in range(0, total, self.batch_size):
            chunk = papers[start:start + self.batch_size]
            tasks = [process_one(start + i, p) for i, p in enumerate(chunk)]
            await asyncio.gather(*tasks)
        
        elapsed = time.time() - started
        print(f"\n[done] Completed in {elapsed:.1f}s\n")
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Verify dataset availability for AI creativity papers with Gemini"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model name (default: gemini-2.5-flash)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel requests (default: 10)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--seed", type=int, default=99, help="Random seed (default: 99)")
    parser.add_argument("--progress-every", type=int, default=25, help="Progress update interval (default: 25)")
    parser.add_argument("--checkpoint-file", type=Path, default=None, help="Checkpoint file for resuming")
    
    args = parser.parse_args()
    
    # Load papers
    print(f"[load] Reading {args.input}")
    papers = load_papers(args.input)
    print(f"[load] Found {len(papers)} papers")
    
    # Filter to only included papers
    papers = [p for p in papers if p.get("include") is True]
    print(f"[load] Filtering to {len(papers)} included papers\n")
    
    # Setup checkpoint
    checkpoint = args.checkpoint_file or args.output.with_stem(f"{args.output.stem}_checkpoint")
    if not checkpoint.exists():
        checkpoint.touch()
        print(f"[init] Created checkpoint: {checkpoint}\n")
    else:
        print(f"[init] Using checkpoint: {checkpoint}\n")
    
    # Verify papers
    verifier = GeminiDatasetVerifier(
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        model=args.model,
        temperature=args.temperature,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_file=checkpoint,
        progress_every=args.progress_every,
    )
    
    results = asyncio.run(verifier.verify_batch(papers))
    
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