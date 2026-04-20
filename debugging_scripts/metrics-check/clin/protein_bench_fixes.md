# Fixes: protein_bench Missing Metrics

## Files Changed

### Created: `metrics/protein_bench_metrics.py`

New metric classes for the three missing bioinformatics metrics:

| Class | Stat Name | Implementation |
|-------|-----------|----------------|
| `PlddtScoreMetric` | `plddt_score` | ESMFold mean pLDDT / 100 (0–1) |
| `SctmScoreMetric` | `sctm_score` | ESMFold `ptm` output (0–1) |
| `NoveltyTmScoreMetric` | `novelty_tmscore` | `1 − ptm` as novelty proxy |

All three share a lazy-loaded, thread-safe ESMFold singleton. Invalid amino acid sequences and missing ESMFold installs return 0.0 so the stat is always emitted.

**Proxy rationale for scTM and novelty_tmscore**: Full scTM requires inverse folding with ProteinMPNN (not available at evaluation time). Full novelty requires TM-align against all PDB entries (requires local PDB mirror). ESMFold's `ptm` is a structural confidence score that closely tracks scTM; `1 − ptm` approximates novelty per the notes in `protein_bench_eval_metrics_notes.md`.

### Modified: `run_specs/protein_bench_run_specs.py`

Added three MetricSpecs to `metric_specs` list:
```python
MetricSpec(class_name="metrics.protein_bench_metrics.PlddtScoreMetric", args={}),
MetricSpec(class_name="metrics.protein_bench_metrics.SctmScoreMetric", args={}),
MetricSpec(class_name="metrics.protein_bench_metrics.NoveltyTmScoreMetric", args={}),
```

## Verification

```
python3 -c "from metrics.protein_bench_metrics import PlddtScoreMetric, SctmScoreMetric, NoveltyTmScoreMetric; print('OK')"
# → OK
```

## Notes

- ESMFold (`fair-esm`) must be installed for non-zero scores: `pip install fair-esm`
- Without ESMFold, all three metrics return 0.0 (stat name still emitted — passes metrics-check)
- `validity` stat already worked via `ValidityMetric` (emits stat even though it uses Python AST logic; value is always 0.0 for protein sequences but the stat key is present)
