"""
HELM Scenario: ProteinBench — De Novo Protein Sequence Design

Paper: "ProteinBench: A Holistic Evaluation of Protein Foundation Models"
       (arXiv:2409.06744, ICLR 2025)
Website: https://proteinbench.github.io/
Authors: ByteDance Research

Task: Given a functional requirement and target length, design a de novo protein
sequence using standard single-letter amino acid codes. Tests whether LLMs can
generate structurally plausible, functionally relevant protein sequences — a
creative design challenge at the boundary of language and molecular biology.

This adapts ProteinBench's "Sequence Design" subtask for general LLM evaluation.
The original benchmark evaluates specialized protein foundation models
(ProteinMPNN, RFdiffusion, ESM3, etc.); this scenario applies the same creative
design challenge to general-purpose language models.

Prompt (adapted from ProteinBench task specification):
  "You are a protein design expert. Design a de novo protein sequence of exactly
   {length} amino acids that {function}. Output only the amino acid sequence using
   standard single-letter codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S,
   T, V, W, Y). Do not include spaces, line breaks, or any other characters."

Dataset: Programmatically generated (no download required).
  4 length categories × 20 functional targets = 80 instances.
  Length categories: 100, 200, 300, 500 residues (matching ProteinBench protocol).
  Functional targets span diverse structural and functional protein classes.

Evaluation: custom (bioinformatics metrics required; see metric_notes.md)
  - Validity:  fraction using only standard 20 amino acid codes + correct length
  - pLDDT:     per-residue confidence from ESMFold (structural plausibility)
  - scTM:      TM-score of ESMFold structure prediction (fold quality)
  - Novelty:   max TM-score vs. PDB entries (lower = more novel design)
  - Diversity: pairwise TM-scores across sequences per (length, function) group

Fields used:   programmatically generated (length, function)
Fields skipped: N/A
Prompt source: Adapted from ProteinBench Sequence Design task specification
               (arXiv:2409.06744, Section 3.1)
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

# Length categories matching ProteinBench's Sequence Design protocol
_LENGTHS = [100, 200, 300, 500]

# 20 diverse functional targets spanning structural classes and functions
_FUNCTIONAL_TARGETS = [
    "folds into a stable monomeric alpha-helical bundle",
    "adopts a stable beta-barrel topology",
    "is thermostable and retains activity above 80°C",
    "binds double-stranded DNA non-specifically as a structural scaffold",
    "is intrinsically disordered under physiological conditions",
    "forms a stable homodimer through a coiled-coil interface",
    "has antimicrobial peptide characteristics with membrane-disrupting activity",
    "contains a metal-binding site coordinating zinc through Cys and His residues",
    "functions as a stable enzyme scaffold with an exposed active site cavity",
    "has high solubility and low aggregation propensity in aqueous buffer",
    "adopts a TIM-barrel (triosephosphate isomerase barrel) fold",
    "is a compact miniprotein with a well-packed hydrophobic core",
    "has beta-sheet packing reminiscent of an immunoglobulin domain",
    "contains a leucine zipper motif enabling specific dimerization",
    "functions as a de novo-designed molecular chaperone scaffold",
    "forms a stable all-alpha topology with at least four helices",
    "is an all-beta protein with anti-parallel strand arrangement",
    "has a Rossman-fold for cofactor (NAD+) binding",
    "adopts a four-helix bundle architecture for protein-protein interaction",
    "is designed as a heat-shock protein mimic for substrate binding",
]

_SYSTEM_CONTEXT = (
    "You are a protein design expert with deep knowledge of structural biology, "
    "protein folding, and molecular function."
)

_PROMPT_TEMPLATE = (
    "{system}\n\n"
    "Design a de novo protein sequence of exactly {length} amino acids that "
    "{function}. The sequence should be novel (not directly copied from known "
    "proteins) and predicted to fold stably.\n\n"
    "Output only the amino acid sequence using standard single-letter codes "
    "(A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y). "
    "Do not include spaces, line breaks, labels, or any other characters."
)


class ProteinBenchScenario(Scenario):
    """
    ProteinBench De Novo Sequence Design — evaluate LLMs on creative protein
    sequence generation.

    Given a target length and functional description, the model must output an
    amino acid sequence (single-letter codes only) of exactly the requested length.
    80 instances: 4 length categories × 20 functional targets.

    Evaluation requires bioinformatics tools (ESMFold/AlphaFold2 for pLDDT/scTM,
    TM-align for novelty vs. PDB). See scenarios/protein_bench/metric_notes.md
    for the full evaluation pipeline.
    """

    name = "protein_bench"
    description = "arXiv:2409.06744 (ProteinBench, ByteDance / ICLR 2025)"
    tags = ["creativity", "protein_design", "scientific_creativity", "open_ended_generation"]

    def get_instances(self, output_path: str) -> List[Instance]:
        instances = []

        for length in _LENGTHS:
            for function in _FUNCTIONAL_TARGETS:
                prompt = _PROMPT_TEMPLATE.format(
                    system=_SYSTEM_CONTEXT,
                    length=length,
                    function=function,
                )

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=[],   # No gold reference; evaluated by bioinformatics pipeline
                        split=TEST_SPLIT,
                    )
                )

        return instances  # 80 instances: 4 lengths × 20 functional targets
