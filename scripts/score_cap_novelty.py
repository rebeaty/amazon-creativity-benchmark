#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-HELM novelty aggregator for CAP (Creativity Assessment Platform) runs.

Ports [human/uva_pilot/scripts/old/score_novelty.py] (the semantic-distance
scorer used on the UVA pilot / Study 3 data) to read HELM's
``benchmark_output/runs/<suite>/cap_*_model=*/scenario_state.json``
and compute the same two metrics per (model, task, item):

  - within_diversity:     avg pairwise cosine distance among a model's own
                          ideas on this item. Only meaningful for multi-
                          response tasks (AUT, SCTT, Design) where a single
                          generation produces multiple semicolon-separated
                          ideas. None for Metaphor / Story.

  - mean_dist_from_pool:  average cosine distance of the model's ideas
                          from the centroid of ALL responses on this item
                          across the current corpus (see caveat below).

Embeddings: google/gemini-embedding-001 (the same model
score_novelty.py uses).

Corpus caveat (matches what we told Roger): the ideal "pool" for novelty
is the full 200-model ABC benchmark corpus, which does not exist yet.
For now the pool defaults to all models run here — this is a proxy. When
the 200-model corpus is available, supply it via --extra-pool-csv to merge.

Output:
  <output_dir>/cap_novelty_by_item.csv     (per (model, task, item))
  <output_dir>/cap_novelty_by_model_task.csv  (aggregated mean per (model, task))

Usage:
    python scripts/score_cap_novelty.py \\
        --runs-root benchmark_output/runs \\
        --output-dir cap_novelty_2026-04-21 \\
        [--extra-pool-csv <future_abc_corpus.csv>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Install google-generativeai: pip install google-generativeai")


EMBEDDING_MODEL = "models/gemini-embedding-001"
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 0.05

MULTI_RESPONSE_TASKS = {"AUT", "SCTT", "Design"}
SINGLE_RESPONSE_TASKS = {"Metaphor", "Story"}

DIRNAME_RE = re.compile(
    r"^cap_(?P<task>aut|sctt|design|metaphor|story),?model=(?P<model_safe>.+)$"
)


def split_ideas(response: str, task: str) -> List[str]:
    """Split multi-response into individual ideas; keep whole for single-response."""
    text = (response or "").strip()
    if not text:
        return []
    if task in MULTI_RESPONSE_TASKS:
        # Primary: semicolons (our scenario asks for semicolon-separated).
        if ";" in text:
            ideas = [i.strip(" .,-") for i in text.split(";")]
        # Fallback: commas if the model ignored the separator request.
        elif text.count(",") >= 2:
            ideas = [i.strip(" .-") for i in text.split(",")]
        else:
            # Models sometimes number ideas without the separator.
            lines = [ln.strip(" .,-*•\t") for ln in text.split("\n") if ln.strip()]
            ideas = lines if len(lines) >= 2 else [text]
        return [i for i in ideas if len(i) >= 3]
    else:
        return [text]


def walk_cap_runs(runs_root: Path):
    """Yield dicts per (suite, model, task, item, response, rep)."""
    for suite_dir in sorted(runs_root.iterdir()):
        if not suite_dir.is_dir() or suite_dir.name.startswith("_"):
            continue
        for run_dir in sorted(suite_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = DIRNAME_RE.match(run_dir.name.lower())
            if not m:
                continue
            task_raw = m.group("task")
            task_map = {"aut": "AUT", "sctt": "SCTT", "design": "Design",
                        "metaphor": "Metaphor", "story": "Story"}
            task = task_map[task_raw]
            model = m.group("model_safe").replace("_", "/", 1)
            ss_path = run_dir / "scenario_state.json"
            if not ss_path.exists():
                continue
            try:
                state = json.loads(ss_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [warn] cannot read {ss_path}: {e}", file=sys.stderr)
                continue
            for rs in state.get("request_states", []):
                inst = rs.get("instance", {})
                xd = inst.get("extra_data", {}) or {}
                prompt_id = xd.get("prompt_id")
                rep = xd.get("rep", 0)
                resp = (rs.get("result", {}) or {}).get("completions", [{}])
                text = (resp[0].get("text", "") if resp else "")
                if prompt_id is None or not text:
                    continue
                yield {
                    "suite": suite_dir.name,
                    "model": model,
                    "task": task,
                    "prompt_id": int(prompt_id),
                    "rep": int(rep),
                    "response": text,
                }


def embed_batch(client, texts: List[str], cache: Dict[str, np.ndarray]):
    """Embed uncached texts. Populates cache in place."""
    uncached = [t for t in texts if t and t not in cache]
    for i in range(0, len(uncached), BATCH_SIZE):
        batch = uncached[i:i + BATCH_SIZE]
        try:
            result = client.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="semantic_similarity",
            )
            embs = result["embedding"]
            for text, emb in zip(batch, embs):
                cache[text] = np.array(emb)
        except Exception as e:
            print(f"  [warn] embedding batch error at offset {i}: {e}", file=sys.stderr)
            # Fallback one at a time
            for t in batch:
                try:
                    r = client.embed_content(
                        model=EMBEDDING_MODEL, content=t,
                        task_type="semantic_similarity",
                    )
                    cache[t] = np.array(r["embedding"])
                except Exception as e2:
                    print(f"    [warn] failed single embed: {e2}", file=sys.stderr)
                    cache[t] = None  # sentinel
        time.sleep(RATE_LIMIT_DELAY)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(an, bn))


def within_entity_diversity(embs: List[Optional[np.ndarray]]) -> Optional[float]:
    valid = [e for e in embs if e is not None]
    if len(valid) <= 1:
        return None
    d = [cosine_distance(a, b) for a, b in combinations(valid, 2)]
    return float(np.mean(d))


def distance_from_centroid(entity: List[Optional[np.ndarray]],
                           pool: List[Optional[np.ndarray]]):
    ve = [e for e in entity if e is not None]
    vp = [e for e in pool if e is not None]
    if not ve or not vp:
        return None, None
    centroid = np.mean(vp, axis=0)
    dists = [cosine_distance(e, centroid) for e in ve]
    return float(np.mean(dists)), float(np.max(dists))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="benchmark_output/runs")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--extra-pool-csv", default=None,
                    help="CSV with additional pool rows (task, prompt_id, response) — "
                         "use to merge the eventual 200-model ABC corpus.")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY required")
    genai.configure(api_key=api_key)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CAP responses from HELM runs
    rows = list(walk_cap_runs(Path(args.runs_root)))
    print(f"Loaded {len(rows)} (model, task, item, rep) cells from HELM runs", file=sys.stderr)
    if not rows:
        print("No CAP run dirs found.", file=sys.stderr)
        return

    # Index: task -> prompt_id -> model -> [ideas]
    task_item_model: Dict[str, Dict[int, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    task_item_pool: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ideas = split_ideas(r["response"], r["task"])
        task_item_model[r["task"]][r["prompt_id"]][r["model"]].extend(ideas)
        task_item_pool[r["task"]][r["prompt_id"]].extend(ideas)

    # Optional extra pool
    if args.extra_pool_csv:
        extra_path = Path(args.extra_pool_csv)
        if extra_path.exists():
            with open(extra_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                n_extra = 0
                for r in reader:
                    t = r.get("task")
                    if t not in MULTI_RESPONSE_TASKS | SINGLE_RESPONSE_TASKS:
                        continue
                    try:
                        pid = int(r["prompt_id"])
                    except Exception:
                        continue
                    ideas = split_ideas(r.get("response", ""), t)
                    task_item_pool[t][pid].extend(ideas)
                    n_extra += len(ideas)
                print(f"Added {n_extra} extra pool ideas from {extra_path}", file=sys.stderr)

    # Collect unique texts and embed
    all_texts = set()
    for t in task_item_pool:
        for pid in task_item_pool[t]:
            all_texts.update(task_item_pool[t][pid])
    print(f"Embedding {len(all_texts)} unique texts …", file=sys.stderr)
    cache: Dict[str, np.ndarray] = {}
    text_list = list(all_texts)
    client = genai
    for i in range(0, len(text_list), BATCH_SIZE):
        embed_batch(client, text_list[i:i + BATCH_SIZE], cache)
        if i % (BATCH_SIZE * 5) == 0 and i > 0:
            print(f"  ... embedded {i}/{len(text_list)}", file=sys.stderr)

    # Compute per (model, task, item)
    per_item = []
    for task in sorted(task_item_model):
        for pid in sorted(task_item_model[task]):
            pool_texts = task_item_pool[task][pid]
            pool_embs = [cache.get(t) for t in pool_texts]
            for model in task_item_model[task][pid]:
                ideas = task_item_model[task][pid][model]
                entity_embs = [cache.get(t) for t in ideas]
                wdiv = within_entity_diversity(entity_embs) if task in MULTI_RESPONSE_TASKS else None
                mean_d, max_d = distance_from_centroid(entity_embs, pool_embs)
                per_item.append({
                    "model": model,
                    "task": task,
                    "prompt_id": pid,
                    "n_ideas": len(ideas),
                    "within_diversity": wdiv if wdiv is not None else "",
                    "mean_dist_from_pool": mean_d if mean_d is not None else "",
                    "max_dist_from_pool": max_d if max_d is not None else "",
                })

    item_csv = out_dir / "cap_novelty_by_item.csv"
    fields = ["model", "task", "prompt_id", "n_ideas",
              "within_diversity", "mean_dist_from_pool", "max_dist_from_pool"]
    with open(item_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_item:
            w.writerow(r)
    print(f"Wrote {item_csv}", file=sys.stderr)

    # Aggregate per (model, task)
    agg: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in per_item:
        k = (r["model"], r["task"])
        for col in ("within_diversity", "mean_dist_from_pool", "max_dist_from_pool"):
            if r[col] != "":
                agg[k][col].append(float(r[col]))
        agg[k]["n_ideas"].append(int(r["n_ideas"]))

    task_csv = out_dir / "cap_novelty_by_model_task.csv"
    agg_fields = ["model", "task", "mean_within_diversity",
                  "mean_dist_from_pool", "mean_max_dist_from_pool", "mean_n_ideas"]
    with open(task_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for (model, task), metrics in sorted(agg.items()):
            w.writerow({
                "model": model,
                "task": task,
                "mean_within_diversity": np.mean(metrics["within_diversity"]) if metrics["within_diversity"] else "",
                "mean_dist_from_pool": np.mean(metrics["mean_dist_from_pool"]) if metrics["mean_dist_from_pool"] else "",
                "mean_max_dist_from_pool": np.mean(metrics["max_dist_from_pool"]) if metrics["max_dist_from_pool"] else "",
                "mean_n_ideas": np.mean(metrics["n_ideas"]) if metrics["n_ideas"] else "",
            })
    print(f"Wrote {task_csv}", file=sys.stderr)
    print(f"\nDone. Cached embeddings: {len(cache)}", file=sys.stderr)


if __name__ == "__main__":
    main()
