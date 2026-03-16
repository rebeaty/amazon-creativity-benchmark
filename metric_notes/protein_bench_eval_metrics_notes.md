# Metric Notes: ProteinBench — De Novo Sequence Design

Source: arXiv:2409.06744 (ICLR 2025); https://proteinbench.github.io/

## Original Evaluation Dimensions (from paper, Section 3.1)

ProteinBench evaluates generated sequences across four dimensions:

| Dimension | Description | Tools Required |
|-----------|-------------|----------------|
| **Quality** | Structural plausibility of generated sequences | ESMFold or AlphaFold2 |
| **Novelty** | Dissimilarity from known proteins in PDB | TM-align vs. PDB |
| **Diversity** | Variation across multiple generated sequences | TM-align pairwise |
| **Robustness** | Stability across prompts and random seeds | Multiple runs |

## Quality Metrics

Run ESMFold (or AlphaFold2) on each generated sequence to predict its 3D structure, then compute:

- **pLDDT**: Per-residue Local Distance Difference Test score (0–100)
  - Average pLDDT > 70 = confidently predicted structure
  - Used as proxy for structural stability/plausibility
- **scTM** (self-consistency TM-score): TM-score of ESMFold structure vs. reference
  - For de novo design: compare predicted structure to idealized fold
  - Range: 0–1; higher = more consistent folding prediction
- **scRMSD** (self-consistency RMSD): RMSD of backbone Cα atoms
  - Lower = better structural quality

## Novelty Metric

Compare each generated sequence's ESMFold-predicted structure against all PDB entries using TM-align:

```
Novelty = 1 - max(TM-score against all PDB structures)
```

- Novelty > 0.5 = structurally distinct from known proteins
- Paper threshold: sequences with max TM-score < 0.5 considered novel

## Diversity Metric

For multiple sequences generated under the same prompt (length + function):

```
Diversity = 1 - mean(pairwise TM-scores between generated sequences)
```

Compute with multiple inference runs (e.g., 10 samples per instance with temperature > 0).

## Validity Metric (additional, for LLM evaluation)

Since LLMs may output invalid characters or wrong lengths:

```python
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def validity(sequence, expected_length):
    seq = sequence.strip().upper()
    return (
        len(seq) == expected_length and
        all(c in VALID_AA for c in seq)
    )
```

Report: fraction of outputs that are valid amino acid sequences of the correct length.

## Evaluation Pipeline

```bash
# Step 1: Extract model outputs from HELM results
# Step 2: Filter for valid sequences (validity check above)
# Step 3: Run ESMFold on valid sequences
esm-fold -i sequences.fasta -o structures/ --num-recycles 3
# Step 4: Compute pLDDT from ESMFold outputs (in JSON metadata)
# Step 5: Run TM-align vs. PDB for novelty
# Step 6: Run pairwise TM-align for diversity

# Python dependencies:
pip install esm  # fair-esm for ESMFold
pip install TMtools  # Python wrapper for TM-align
```

## Recommended Tools

| Tool | Purpose | Install |
|------|---------|---------|
| ESMFold (Meta) | Structure prediction | `pip install esm` |
| AlphaFold2 | Structure prediction | ColabFold recommended |
| TM-align | Structure comparison | https://zhanggroup.org/TM-align/ |
| TMtools | Python TM-align wrapper | `pip install TMtools` |
| BioPython | Sequence handling | `pip install biopython` |

## Baseline Context (from paper)

Original benchmark results for specialized protein models on Sequence Design:

| Model | pLDDT | scTM | Novelty | Diversity |
|-------|-------|------|---------|-----------|
| ProteinMPNN | ~80 | ~0.85 | ~0.45 | ~0.70 |
| ESM3 | ~82 | ~0.87 | ~0.40 | ~0.72 |
| RFdiffusion | ~85 | ~0.88 | ~0.50 | ~0.75 |

LLM baseline performance expected to be substantially lower on pLDDT/scTM, but
potentially higher on novelty (LLMs may generate more unusual sequences).

## Notes

- The original benchmark does NOT evaluate general LLMs; these metrics provide
  a cross-domain comparison point
- For HELM integration, implement as a custom metric class that calls ESMFold
  via subprocess and parses output JSON
- Diversity requires multiple samples per instance (set `num_outputs > 1` in
  AdapterSpec); default HELM evaluation uses 1 output
- Consider reporting validity as a primary metric since it requires no
  bioinformatics tools
