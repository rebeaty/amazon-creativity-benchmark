"""Aggregate per-benchmark faithfulness review YAMLs into a single Markdown report."""
import os
import glob
import sys
from collections import defaultdict, Counter
from datetime import date

import yaml

REVIEW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithfulness_review")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FAITHFULNESS_REPORT.md")


def _norm_verdict(v: str) -> str:
    if not v:
        return "UNKNOWN"
    v = v.strip().upper()
    # Normalise the verdict variants the agents invented.
    if v in {"FAITHFUL"}:
        return "FAITHFUL"
    if v in {"MOSTLY_FAITHFUL", "PARTIALLY_FAITHFUL", "PARTIAL", "PARTIAL_FAITHFUL", "FAITHFUL_WITH_CAVEATS"}:
        return "MOSTLY_FAITHFUL"
    if v in {"DIVERGENT", "UNFAITHFUL", "NOT_FAITHFUL", "METRIC_MISMATCH"}:
        return "DIVERGENT"
    return v


def _as_str(x):
    if x is None:
        return ""
    return str(x)


def _norm_bool(x):
    if x is True:
        return "true"
    if x is False:
        return "false"
    return str(x) if x is not None else "unknown"


def main():
    files = sorted(glob.glob(os.path.join(REVIEW_DIR, "*.yaml")))
    reviews = []
    parse_errors = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as e:
            parse_errors.append((name, str(e)))
            continue
        if not isinstance(data, dict):
            parse_errors.append((name, f"non-dict YAML: {type(data).__name__}"))
            continue
        data["_name"] = data.get("name", name)
        data["_verdict"] = _norm_verdict(_as_str(data.get("verdict")))
        reviews.append(data)

    # Counts
    verdict_counts = Counter(r["_verdict"] for r in reviews)

    # Blocking fixes across all reviews
    blocking = []
    for r in reviews:
        fixes = r.get("fixes_before_run") or []
        if not isinstance(fixes, list):
            continue
        for fx in fixes:
            if isinstance(fx, dict):
                sev = _as_str(fx.get("severity", "")).lower()
                if sev.startswith("blocking"):
                    blocking.append((r["_name"], _as_str(fx.get("action", "")).strip()))

    # Partition by verdict
    buckets = defaultdict(list)
    for r in reviews:
        buckets[r["_verdict"]].append(r)

    lines = []
    lines.append(f"# Faithfulness Review — {date.today().isoformat()}")
    lines.append("")
    lines.append(
        f"Automated per-benchmark audit of the **{len(reviews)} benchmarks** that produced "
        "`stats.json` in any suite. Each review compares the scenario prompt / run_spec metrics / "
        "judge config / sample predictions against the source paper's intended evaluation, using "
        "`scenarios/`, `run_specs/`, `metric_notes/`, and `benchmark_output/runs/` as evidence, "
        "with WebFetch attempted on the paper URL from each scenario docstring. Reviewer was "
        "Claude general-purpose subagent, one invocation per benchmark."
    )
    lines.append("")
    lines.append("**Verdict scale** (normalised; agents used a few synonyms):")
    lines.append("")
    lines.append("- **FAITHFUL** — matches the paper to a reasonable standard")
    lines.append("- **MOSTLY_FAITHFUL** — minor drift (prompt wording, extra metrics) but valid to run")
    lines.append("- **DIVERGENT** — major drift that would compromise the evaluation; should fix before the full run")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Verdict | Count | % of reviewed |")
    lines.append("|---|---:|---:|")
    for v in ("FAITHFUL", "MOSTLY_FAITHFUL", "DIVERGENT", "UNKNOWN"):
        n = verdict_counts.get(v, 0)
        if n == 0 and v == "UNKNOWN":
            continue
        pct = 100.0 * n / max(1, len(reviews))
        lines.append(f"| {v} | {n} | {pct:.1f}% |")
    lines.append(f"| **TOTAL** | **{len(reviews)}** | 100.0% |")
    lines.append("")
    if parse_errors:
        lines.append(f"_{len(parse_errors)} YAML files failed to parse and are excluded from the tally:_")
        for name, err in parse_errors:
            lines.append(f"  - `{name}`: {err[:120]}")
        lines.append("")

    # Blocking fixes at the top — this is what Roger needs most
    if blocking:
        lines.append(f"## Blocking fixes before launch ({len(blocking)} total)")
        lines.append("")
        lines.append("These were flagged as `severity: blocking` by the reviewer — running the full "
                     "evaluation against these without patching first will produce metrics that "
                     "don't represent what the source paper measures. Grouped by benchmark:")
        lines.append("")
        by_bench = defaultdict(list)
        for name, action in blocking:
            by_bench[name].append(action)
        for name in sorted(by_bench):
            lines.append(f"### `{name}`")
            for action in by_bench[name]:
                lines.append(f"- {action}")
            lines.append("")

    # Per-verdict sections with per-benchmark one-liners
    for verdict in ("DIVERGENT", "MOSTLY_FAITHFUL", "FAITHFUL", "UNKNOWN"):
        rows = buckets.get(verdict, [])
        if not rows:
            continue
        lines.append(f"## {verdict} ({len(rows)})")
        lines.append("")
        lines.append("| Benchmark | Prompt | Data | Metric | Judge | Output | Paper |")
        lines.append("|---|:-:|:-:|:-:|:-:|:-:|---|")
        for r in sorted(rows, key=lambda x: x["_name"]):
            checks = r.get("checks") or {}
            if not isinstance(checks, dict):
                checks = {}
            def c(k):
                return _norm_bool(checks.get(k))[:7]
            paper = _as_str(r.get("paper_url", "")).strip() or "—"
            paper_disp = paper if len(paper) < 60 else paper[:57] + "…"
            lines.append(
                f"| `{r['_name']}` | {c('prompt_matches')} | {c('data_split_correct')} | "
                f"{c('metric_matches')} | {c('judge_config_matches')} | {c('output_shape_sane')} | "
                f"{paper_disp} |"
            )
        lines.append("")

        # Detail block per-benchmark: notes + gaps
        for r in sorted(rows, key=lambda x: x["_name"]):
            name = r["_name"]
            notes = _as_str(r.get("notes", "")).strip()
            gaps = r.get("gaps") or []
            if not isinstance(gaps, list):
                gaps = [str(gaps)]
            lines.append(f"### `{name}`")
            if notes:
                lines.append(f"_{notes}_")
                lines.append("")
            if gaps:
                lines.append("**Gaps:**")
                for g in gaps:
                    lines.append(f"- {str(g).strip()}")
                lines.append("")

    # Write
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {OUT_PATH}")
    print(f"Total reviewed: {len(reviews)}")
    print(f"Verdict counts: {dict(verdict_counts)}")
    print(f"Blocking fixes: {len(blocking)}")
    if parse_errors:
        print(f"Parse errors: {len(parse_errors)}")


if __name__ == "__main__":
    main()
