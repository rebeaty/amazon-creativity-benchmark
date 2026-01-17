#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Literature Harvester for AI Creativity Benchmarks

Harvests papers from Semantic Scholar Graph API v1 (BULK SEARCH endpoint) 
for a systematic literature review on AI creativity benchmarks (2018-2025).

Features:
  - Deterministic checkpoint-based resume capability
  - Token-based pagination (BULK endpoint)
  - Rate limiting with exponential backoff and jitter
  - Streaming JSONL + CSV output as results arrive
  - Configurable result limits and delays

Requirements:
  Python 3.10+
  urllib3>=2.0

Installation:
  pip install 'urllib3>=2.0'

Quick Start:
  python lit_harvester_s2.py run --out out/harvest

Environment:
  SEMANTIC_SCHOLAR_API_KEY: Optional for higher rate limits (1 req/sec vs 100 req/5min)

Output Files:
  out/harvest.jsonl - Newline-delimited papers
  out/harvest.csv - Spreadsheet format
  out/harvest__checkpoint.json - Resume point (tracks seen papers)

"""

import argparse
import csv
import dataclasses as dc
import datetime as dt
import json
import os
import random
import re
import sys
import time
import typing as T
from pathlib import Path
from urllib.parse import quote_plus

import urllib.request
import urllib.error

# Configuration based on PRISMA protocol
DEFAULT_SINCE = "2018-01-01"
DEFAULT_UNTIL = "2025-12-31"
DEFAULT_SEED = 99

AI_TERMS = [
    'LLM*', 'large language model*', 'vision-language model*', 'VLM*',
    'foundation model*', 'multimodal model*', '"generative AI"', 'diffusion model*'
]

CREATIVITY_TERMS = [
    'creativ*', '"divergent thinking"', '"creative problem solving"',
    '"story generation"', '"scientific creativity"', 'originality', 'novelty',
    'ideation', '"idea generation"', '"hypothesis generation"',
    'metaphor*', 'analogy', 'humor', 'joke*', 'poetry', 'poem*', '"design thinking"'
]

EVAL_TERMS = [
    'benchmark*', 'dataset*', '"shared task*"', 'leaderboard', '"evaluation framework"',
    'metric*', '"model judge*"', 'scoring', 'rubric', '"human evaluation"',
    '"user study"', 'psychometric*'
]


def _or_block(terms: list[str]) -> str:
    """Create OR block using S2 BULK syntax: (term1 | term2 | term3)"""
    return '(' + ' | '.join(terms) + ')'


S2_QUERY = f"{_or_block(AI_TERMS)} + {_or_block(CREATIVITY_TERMS)} + {_or_block(EVAL_TERMS)}"


@dc.dataclass
class Paper:
    source: str
    title: str
    abstract: str
    year: int | None
    published: str | None
    authors: list[str]
    venue: str | None
    doi: str | None
    arxiv_id: str | None
    url: str | None
    pdf_url: str | None
    s2_paper_id: str | None
    categories: list[str]

    def dedupe_key(self) -> str:
        if self.doi:
            return f"doi::{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv::{self.arxiv_id.lower()}"
        base = re.sub(r"\s+", " ", (self.title or "").strip().lower())
        return f"titleyear::{base}::{self.year or 0}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sleep_with_jitter(seconds: float) -> None:
    time.sleep(seconds + random.uniform(0, seconds * 0.15))


@dc.dataclass
class S2Config:
    base_url: str = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    page_size: int = 100
    max_retries: int = 6
    min_delay_sec: float = 1.0


def s2_search(query: str, since: str, until: str,
              max_results: int | None, cfg: S2Config) -> T.Iterable[Paper]:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        print("[s2] WARNING: No API key. Rate limits apply (~100 req/5min shared pool)")
        print("[s2] Set SEMANTIC_SCHOLAR_API_KEY env var for higher limits (1 req/sec)")
    
    fetched = 0
    token = None

    if len(query) > 200:
        print(f"[s2] Query: {query[:200]}... (truncated)")
    else:
        print(f"[s2] Query: {query}")

    fields = [
        "title", "abstract", "year", "venue", "authors", "externalIds",
        "url", "isOpenAccess", "openAccessPdf", "publicationTypes",
    ]

    while True:
        if max_results is not None and fetched >= max_results:
            break
            
        size = min(cfg.page_size, (max_results - fetched) if max_results else cfg.page_size)
        params = {
            "query": query,
            "limit": str(size),
            "fields": ",".join(fields),
        }
        if token:
            params["token"] = token
            
        url = cfg.base_url + "?" + "&".join(f"{k}={quote_plus(v)}" for k, v in params.items())

        attempt = 0
        print(f"[s2] request limit={size}" + (f" token={token[:20]}..." if token else ""))
        while True:
            try:
                headers = {"User-Agent": f"lit-harvester/{__version__}"}
                if api_key:
                    headers["x-api-key"] = api_key
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read()
                break
            except urllib.error.HTTPError as e:
                attempt += 1
                try:
                    error_body = json.loads(e.read().decode('utf-8'))
                    error_msg = error_body.get('message', e.reason)
                except:
                    error_msg = e.reason
                
                if e.code == 429 and attempt <= cfg.max_retries:
                    retry_after_hdr = e.headers.get("Retry-After") if e.headers else None
                    if retry_after_hdr:
                        try:
                            delay = float(retry_after_hdr)
                        except Exception:
                            delay = cfg.min_delay_sec * (2 ** (attempt - 1))
                    else:
                        delay = cfg.min_delay_sec * (2 ** (attempt - 1))
                    print(f"[s2] 429 rate limit, retry {attempt}, waiting {delay:.1f}s")
                    sleep_with_jitter(delay)
                    continue
                if e.code in (500, 502, 503, 504) and attempt <= cfg.max_retries:
                    delay = cfg.min_delay_sec * (2 ** (attempt - 1))
                    print(f"[s2] HTTP {e.code}, retry {attempt}, waiting {delay:.1f}s")
                    sleep_with_jitter(delay)
                    continue
                print(f"[s2] HTTP error {e.code}: {error_msg}; terminating", file=sys.stderr)
                return
            except Exception as e:
                attempt += 1
                if attempt <= cfg.max_retries:
                    delay = cfg.min_delay_sec * (2 ** (attempt - 1))
                    print(f"[s2] Error {e}, retry {attempt}, waiting {delay:.1f}s")
                    sleep_with_jitter(delay)
                    continue
                print(f"[s2] Error: {e}; terminating", file=sys.stderr)
                return

        sleep_with_jitter(cfg.min_delay_sec)

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            print(f"[s2] JSON parse error: {e}", file=sys.stderr)
            print(f"[s2] Response body (first 500 chars): {body.decode('utf-8')[:500]}", file=sys.stderr)
            return

        data = payload.get("data", [])
        token = payload.get("token")
        total = payload.get("total")
        
        if fetched == 0 and total is not None:
            print(f"[s2] total matching papers: {total}")
        
        if not data:
            print("[s2] no more results")
            break

        page_count = 0
        for d in data:
            year = d.get("year")
            if year is not None:
                try:
                    y = int(year)
                except Exception:
                    y = None
            else:
                y = None
            
            # Apply date filter
            if y:
                since_year = dt.date.fromisoformat(since).year
                until_year = dt.date.fromisoformat(until).year
                if y < since_year or y > until_year:
                    continue

            ext = d.get("externalIds") or {}
            doi = ext.get("DOI")
            arxiv_id = ext.get("ArXiv") or ext.get("ARXIV")
            authors = [a.get("name") for a in (d.get("authors") or []) if a.get("name")]
            url = d.get("url")
            pdf_url = (d.get("openAccessPdf") or {}).get("url")
            paper = Paper(
                source="s2",
                title=(d.get("title") or "").strip(),
                abstract=(d.get("abstract") or "").strip(),
                year=y,
                published=None,
                authors=authors,
                venue=d.get("venue"),
                doi=doi,
                arxiv_id=arxiv_id,
                url=url,
                pdf_url=pdf_url,
                s2_paper_id=d.get("paperId"),
                categories=d.get("publicationTypes") or [],
            )
            yield paper
            page_count += 1
            fetched += 1
            if max_results is not None and fetched >= max_results:
                break

        print(f"[s2] received {page_count} entries (total so far: {fetched})")
        if page_count == 0 or not token:
            break


def run_harvest(out_base: Path, max_results: int | None, delay: float) -> None:
    random.seed(DEFAULT_SEED)
    
    ensure_dir(out_base.parent)
    jsonl_path = out_base.with_suffix(".jsonl")
    csv_path = out_base.with_suffix(".csv")
    checkpoint_path = out_base.parent / (out_base.name + "__checkpoint.json")

    print(f"[init] Literature Harvester v{__version__}")
    print(f"[init] Date range: {DEFAULT_SINCE} to {DEFAULT_UNTIL}")
    print(f"[init] Seed: {DEFAULT_SEED}")
    print(f"[init] max_results={max_results or 'unlimited'}")
    print(f"[init] api_delay={delay}s\n")

    seen: set[str] = set()
    if checkpoint_path.exists():
        try:
            seen = set(json.load(checkpoint_path.open("r", encoding="utf-8")).get("seen", []))
            print(f"[ckpt] loaded {len(seen)} previously seen papers from checkpoint\n")
        except Exception:
            pass

    def dedupe_stream(stream: T.Iterable[Paper]) -> T.Iterable[Paper]:
        for p in stream:
            key = p.dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            yield p

    cfg = S2Config(min_delay_sec=delay)
    stream = s2_search(S2_QUERY, DEFAULT_SINCE, DEFAULT_UNTIL, max_results, cfg)

    fieldnames = [
        "source", "title", "abstract", "year", "published", "authors", "venue",
        "doi", "arxiv_id", "url", "pdf_url", "s2_paper_id", "categories",
    ]
    jsonl_f = jsonl_path.open("w", encoding="utf-8")
    csv_f = csv_path.open("w", encoding="utf-8", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames)
    csv_w.writeheader()
    print(f"[io] JSONL: {jsonl_path}")
    print(f"[io] CSV:   {csv_path}\n")

    def write_one(p: Paper, idx: int) -> None:
        jsonl_f.write(json.dumps(dc.asdict(p), ensure_ascii=False, sort_keys=True) + "\n")
        d = dc.asdict(p)
        d["authors"] = "; ".join(d.get("authors") or [])
        d["categories"] = "; ".join(d.get("categories") or [])
        csv_w.writerow(d)
        if idx % 20 == 0:
            jsonl_f.flush()
            csv_f.flush()

    collected = 0
    for p in dedupe_stream(stream):
        collected += 1
        write_one(p, collected)
        if collected % 25 == 0:
            print(f"[write] collected {collected}: {p.title[:80]}…")
        if collected % 200 == 0:
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump({"seen": sorted(list(seen)), "count": collected}, f, sort_keys=True)

    jsonl_f.flush()
    csv_f.flush()
    jsonl_f.close()
    csv_f.close()

    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump({"seen": sorted(list(seen)), "count": collected}, f, sort_keys=True)

    print(f"\n[done] collected {collected} papers")
    print(f"[done] checkpoint: {checkpoint_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Harvest Semantic Scholar for AI creativity benchmark literature (2018-2025)",
        epilog=f"Version {__version__} - {__license__} License"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run the harvest")
    runp.add_argument("--out", required=True, help="Output base path (no extension)")
    runp.add_argument("--max-results", type=int, default=None, help="Max results to fetch")
    runp.add_argument("--delay", type=float, default=1.0, help="Seconds between API requests")

    qp = sub.add_parser("print-query", help="Print the search query")
    
    vp = sub.add_parser("version", help="Print version information")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    
    if args.cmd == "version":
        print(f"Literature Harvester v{__version__}")
        print(f"Author: {__author__}")
        print(f"License: {__license__}")
        print(f"Date Range: {DEFAULT_SINCE} to {DEFAULT_UNTIL}")
        print(f"Seed: {DEFAULT_SEED}")
        return
        
    if args.cmd == "print-query":
        print("Semantic Scholar BULK Query:")
        print(S2_QUERY)
        print(f"\nDate Range: {DEFAULT_SINCE} to {DEFAULT_UNTIL}")
        print(f"Seed: {DEFAULT_SEED}")
        return
        
    if args.cmd == "run":
        run_harvest(
            out_base=Path(args.out),
            max_results=args.max_results,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()