#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Downloader for Systematic Review Papers

Downloads PDFs for included papers using Gemini + Google Search.
Outputs a manifest of successful downloads and failures.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

try:
    import google.genai as genai
    from google.genai.types import GenerateContentConfig, GoogleSearch, Tool
except ImportError:
    raise ImportError("Install google-genai: pip install google-genai")

try:
    from semanticscholar import SemanticScholar
except ImportError:
    SemanticScholar = None

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_TEMPERATURE = 0.3
DEFAULT_CONCURRENCY = 5

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

PDF_FINDER_PROMPT = """Find the direct PDF download link for this research paper.

PAPER:
Title: {title}
Year: {year}
DOI: {doi}

SEARCH STRATEGY:
1. Try: "{title} pdf" or "{title} filetype:pdf"
2. Check: arXiv, ACL Anthology, author websites, institutional repos
3. Look for direct PDF links (ending in .pdf)

IMPORTANT:
- Return the DIRECT PDF download URL (not landing pages)
- Verify it's the correct paper by checking title/authors
- Prefer: arXiv > ACL Anthology > university repos > ResearchGate
- ArXiv URLs: https://arxiv.org/pdf/XXXX.XXXXX.pdf
- ACL Anthology: https://aclanthology.org/YYYY.venue-X.pdf

RETURN ONLY THIS JSON (no markdown, no explanation):
{{
  "pdf_url": "direct PDF link or null",
  "source": "arxiv/acl/university/researchgate/other/not_found",
  "confidence": "high/medium/low",
  "notes": "Brief explanation"
}}

CRITICAL: Return ONLY the JSON object. No other text."""


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_s2_client() -> Optional["SemanticScholar"]:
    """Initialize Semantic Scholar client if available."""
    if SemanticScholar is None:
        return None
    try:
        return SemanticScholar(api_key=SEMANTIC_SCHOLAR_API_KEY)
    except Exception:
        return None


def resolve_pdf_url_fallback(row: Dict[str, Any], s2_client) -> Optional[str]:
    """Try to construct PDF URL from DOI, paper_id, or other fields."""
    doi = (row.get("doi") or "").strip()
    
    # ArXiv DOI
    if doi:
        arxiv_match = re.search(r'arxiv[.\s:](\d+\.\d+)', doi, re.IGNORECASE)
        if arxiv_match:
            return f"https://arxiv.org/pdf/{arxiv_match.group(1)}.pdf"
        
        # ACL Anthology
        if doi.startswith("10.18653/v1/"):
            anthology_id = doi.replace("10.18653/v1/", "")
            return f"https://aclanthology.org/{anthology_id}.pdf"
    
    # Semantic Scholar
    paper_id = (row.get("paper_id") or "").strip()
    if paper_id and s2_client:
        try:
            paper = s2_client.get_paper(paper_id, fields=["openAccessPdf", "externalIds"])
            
            oap = getattr(paper, "openAccessPdf", None)
            if oap:
                pdf_url = getattr(oap, "url", None) or (oap.get("url") if isinstance(oap, dict) else None)
                if pdf_url:
                    return pdf_url
            
            external_ids = getattr(paper, "externalIds", None)
            if isinstance(external_ids, dict):
                arxiv_id = external_ids.get("ArXiv") or external_ids.get("ARXIV")
                if arxiv_id:
                    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        except Exception:
            pass
    
    # URL columns
    for key in ["pdf_url", "pdf", "openaccess_pdf", "paper_url", "url"]:
        v = row.get(key)
        if isinstance(v, str) and v.strip().startswith("http"):
            url = v.strip()
            if url.endswith(".pdf") or "pdf" in url.lower():
                return url
    
    return None


def parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse Gemini JSON response, handling markdown fences."""
    if not raw:
        return None
    
    text = raw.strip()
    
    # Strip markdown fences
    if "```" in text:
        if "```json" in text:
            text = text.split("```json", 1)[1]
        else:
            text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()
    
    # Find JSON object
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except Exception:
            pass
    
    return None


async def download_pdf(url: str, cache_dir: Path, paper_id: str) -> Optional[Path]:
    """Download PDF to cache directory."""
    # Use paper_id for filename if available, otherwise hash URL
    if paper_id:
        safe_id = "".join(c for c in paper_id if c.isalnum() or c in "-_")[:100]
        pdf_path = cache_dir / f"{safe_id}.pdf"
    else:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        pdf_path = cache_dir / f"{url_hash}.pdf"
    
    # Check cache
    if pdf_path.exists():
        print(f"  ↻ Cached: {pdf_path.name}")
        return pdf_path
    
    # Download
    try:
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
            print(f"  ✓ Downloaded: {pdf_path.name}")
            return pdf_path
    except Exception as e:
        print(f"  ✗ Download failed: {type(e).__name__}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOADER
# ──────────────────────────────────────────────────────────────────────────────

class PDFDownloader:
    def __init__(self, api_key: str, model: str, concurrency: int, cache_dir: Path):
        if not api_key:
            raise ValueError("GEMINI_API_KEY required")
        
        self.client = genai.Client(api_key=api_key).aio
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.cache_dir = cache_dir
        self.s2_client = get_s2_client()
        
        try:
            self.search_tool = Tool(google_search=GoogleSearch())
        except Exception:
            print("⚠ Google Search not available")
            self.search_tool = None
    
    async def find_pdf_url(self, title: str, year: str, doi: str) -> Optional[str]:
        """Use Gemini + Google Search to find PDF URL."""
        if not self.search_tool:
            return None
        
        prompt = PDF_FINDER_PROMPT.format(
            title=title or "Unknown",
            year=year or "Unknown",
            doi=doi or "None"
        )
        
        try:
            async with self.semaphore:
                response = await self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=GenerateContentConfig(
                        tools=[self.search_tool],
                        temperature=GEMINI_TEMPERATURE
                    )
                )
            
            result = parse_json_response(response.text if response else "")
            if result and result.get("pdf_url"):
                url = result["pdf_url"]
                if isinstance(url, str) and url.startswith("http"):
                    return url.strip()
        except Exception as e:
            print(f"  ⚠ Search error: {type(e).__name__}")
        
        return None
    
    async def download_one(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Download PDF for one paper."""
        paper_id = row.get("paper_id", "")
        title = (row.get("title") or "").strip()
        year = (row.get("year") or "").strip()
        doi = (row.get("doi") or "").strip()
        
        # Try Google Search first
        pdf_url = await self.find_pdf_url(title, year, doi)
        if pdf_url:
            print(f"  → Found via search: {pdf_url[:60]}...")
        else:
            # Fallback to URL construction
            pdf_url = resolve_pdf_url_fallback(row, self.s2_client)
            if pdf_url:
                print(f"  → Found via fallback: {pdf_url[:60]}...")
        
        if not pdf_url:
            return {
                "paper_id": paper_id,
                "title": title,
                "status": "no_url",
                "pdf_path": None
            }
        
        # Download
        pdf_path = await download_pdf(pdf_url, self.cache_dir, paper_id)
        
        return {
            "paper_id": paper_id,
            "title": title,
            "status": "success" if pdf_path else "download_failed",
            "pdf_path": str(pdf_path.relative_to(self.cache_dir)) if pdf_path else None,
            "pdf_url": pdf_url
        }
    
    async def download_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Download PDFs for batch of papers."""
        results = []
        total = len(rows)
        
        for i, row in enumerate(rows, 1):
            print(f"\n[{i}/{total}] {row.get('title', 'Untitled')[:70]}...")
            result = await self.download_one(row)
            results.append(result)
        
        return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def load_included_papers(csv_path: str) -> List[Dict[str, Any]]:
    """Load papers marked for inclusion."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    included = []
    for r in rows:
        final = (r.get("final_decision") or "").strip().lower()
        if final == "include":
            included.append(r)
            continue
        
        n_include = int(r.get("n_include") or 0)
        n_exclude = int(r.get("n_exclude") or 0)
        if n_include > 0 and n_exclude == 0:
            included.append(r)
    
    print(f"Loaded {len(included)} included papers from {len(rows)} total")
    return included


async def main_async(args):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n▶ PDF cache: {cache_dir.resolve()}")
    print(f"▶ Concurrency: {args.concurrency}\n")
    
    # Load papers
    papers = load_included_papers(args.input_csv)
    
    # Download
    downloader = PDFDownloader(
        GEMINI_API_KEY,
        GEMINI_MODEL,
        args.concurrency,
        cache_dir
    )
    
    results = await downloader.download_batch(papers)
    
    # Save manifest
    manifest_path = cache_dir / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Summary
    n_success = sum(1 for r in results if r["status"] == "success")
    n_no_url = sum(1 for r in results if r["status"] == "no_url")
    n_failed = sum(1 for r in results if r["status"] == "download_failed")
    
    print(f"\n{'='*70}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Total papers:      {len(results)}")
    print(f"Successfully downloaded: {n_success}")
    print(f"No URL found:      {n_no_url}")
    print(f"Download failed:   {n_failed}")
    print(f"\nManifest saved to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download PDFs for systematic review")
    parser.add_argument("input_csv", help="Input CSV with included papers")
    parser.add_argument("--cache-dir", default="pdf_cache", help="PDF cache directory")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent downloads")
    
    args = parser.parse_args()
    
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
