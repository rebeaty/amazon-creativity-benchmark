"""Reproducible per-evaluation-unit sampling.

Used by scenarios that should return a fixed random subset of items for the
creativity-benchmark analysis. Materializes the sampled index list to
``data/sampling/<unit>_sample.json`` on first use so every model run sees
exactly the same items.

Sampling rules (see plan file):
  - If len(items) < n: return all items unchanged (Case 1).
  - Else: random sample n items using a Random seeded from
    ``RNG_SEED + unit_name`` (Case 2). Materialized JSON records the
    exact indices selected.

Usage in a scenario:

    from scenarios._sample import sampled
    ...
    instances = self._build_all_instances(...)
    return sampled("brainteaser_subtask=sentence_puzzle", instances)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import List, TypeVar

T = TypeVar("T")

_DEFAULT_N = 200
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEED_FILE = _REPO_ROOT / "data" / "sampling" / "rng_seed.txt"
_SAMPLE_DIR = _REPO_ROOT / "data" / "sampling"


def _seed() -> int:
    return int(_SEED_FILE.read_text().strip())


def _unit_seed(unit_name: str) -> int:
    # Stable int from RNG_SEED + unit_name. md5 is fine for deterministic hashing.
    base = _seed()
    h = hashlib.md5(unit_name.encode("utf-8")).hexdigest()
    return (base + int(h[:8], 16)) % (2**31)


def sampled(unit_name: str, items: List[T], n: int = _DEFAULT_N) -> List[T]:
    """Return ≤n items for this evaluation unit, reproducibly.

    If ``data/sampling/<unit>_sample.json`` exists, reuse the indices it
    records (load-only path, safe across runs).
    Otherwise, generate a fresh sample with Random(unit-seed), materialize
    the JSON, then return the subset.
    """
    total = len(items)
    if total == 0:
        return items
    safe_unit = unit_name.replace("/", "_")
    sample_json = _SAMPLE_DIR / f"{safe_unit}_sample.json"

    if sample_json.exists():
        idx = json.loads(sample_json.read_text(encoding="utf-8"))["indices"]
        # Guard: if the dataset size changed under our feet, fall back to the
        # intersection (indices that are still valid) and warn.
        idx = [i for i in idx if 0 <= i < total]
        return [items[i] for i in idx]

    if total <= n:
        indices = list(range(total))
    else:
        rng = random.Random(_unit_seed(unit_name))
        indices = sorted(rng.sample(range(total), n))

    _SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_json.write_text(
        json.dumps({
            "unit": unit_name,
            "total_items": total,
            "sampled": len(indices),
            "indices": indices,
        }, indent=2),
        encoding="utf-8",
    )
    return [items[i] for i in indices]
