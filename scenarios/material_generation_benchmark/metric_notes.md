# Evaluation Metrics: Material Generation Benchmark (MGB)

**Source:** MGB: The Material Generation Benchmark (OpenReview)

## Overview

The Material Generation Benchmark requires **custom domain-specific metrics** to evaluate crystal structure generation. Standard text-based metrics (BLEU, ROUGE, exact match) are insufficient because they cannot assess structural similarity or physical validity of generated CIF files.

## Required Evaluation Dimensions

MGB evaluates generative models across six categories:

### 1. Matching Accuracy (Structural Similarity)
- **Match Rate (MR)**: Percentage of generated structures that match reference structures
  - Uses structural matching algorithms (e.g., pymatgen StructureMatcher)
  - Considers crystallographic symmetry (structures can be equivalent under rotation/translation)
  - Typical threshold: RMSD < 0.5 Å after optimal alignment

- **Root Mean Square Deviation (RMSD)**: Average atomic position deviation
  - Computed after optimal structural alignment
  - Lower is better (0 = perfect match)

### 2. Generation Quality
- **Validity Rate**: Percentage of outputs that are valid CIF files
  - Parse CIF using pymatgen or ASE (Atomic Simulation Environment)
  - Check for valid crystal structure parameters

- **Chemical Validity**: Whether generated composition is chemically reasonable
  - Elements match the specified formula
  - Stoichiometry is correct
  - No unphysical atomic arrangements

- **Diversity**: Structural diversity of generated materials
  - Measured using pairwise RMSD or structural fingerprints
  - Higher diversity indicates broader exploration of structure space

- **Property Distribution Alignment**: How well generated structures match property distributions
  - Compare distributions of formation energy, band gap, density, etc.
  - Use KL divergence or Wasserstein distance

### 3. Out-of-Distribution (OOD) Generation
- **OOD Validity**: Can the model generate valid structures for unseen compositions/topologies?
  - Critical for assessing generalization capability
  - MGB constructs dedicated OOD test sets

- **Novelty Rate**: Percentage of generated structures not in training data
  - Uses structural matching to check against training set

### 4. Physical Plausibility
- **Collision Detection**: Identify structures with overlapping atoms
  - Check minimum interatomic distances
  - Flag structures with atoms too close (< 0.5 Å)

- **Density Check**: Whether material density is physically reasonable
  - Extreme densities (too high/low) indicate unphysical structures

### 5. Symmetry Awareness
- **Space Group Consistency**: Does generated structure match specified space group?
  - Use symmetry detection algorithms (spglib)
  - Compare detected vs. specified space group

- **Translation Symmetry**: Proper periodic boundary conditions
  - Crystal structures must respect periodic symmetry

### 6. Computational Complexity
- **Inference Time**: Time to generate each structure
- **Model Size**: Number of parameters

## Implementation Requirements

### Required Libraries
- **pymatgen**: Crystal structure manipulation and comparison
- **ASE (Atomic Simulation Environment)**: Alternative structure handling
- **spglib**: Space group detection and symmetry operations
- **numpy/scipy**: Numerical computations

### Example Evaluation Pipeline

```python
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

def evaluate_structure_generation(generated_cif: str, reference_cif: str):
    """
    Evaluate generated crystal structure against reference.

    Returns dict with metrics:
    - is_valid: bool (can parse CIF)
    - matches: bool (structure matches reference)
    - rmsd: float (structural deviation)
    - has_collisions: bool (overlapping atoms)
    - space_group_match: bool (symmetry preserved)
    """
    try:
        gen_struct = Structure.from_str(generated_cif, fmt="cif")
        ref_struct = Structure.from_str(reference_cif, fmt="cif")

        # Structural matching
        matcher = StructureMatcher()
        matches = matcher.fit(gen_struct, ref_struct)
        rmsd = matcher.get_rms_dist(gen_struct, ref_struct)[0] if matches else float('inf')

        # Validity checks
        has_collisions = check_atomic_collisions(gen_struct)
        sg_match = check_space_group(gen_struct, ref_struct)

        return {
            'is_valid': True,
            'matches': matches,
            'rmsd': rmsd,
            'has_collisions': has_collisions,
            'space_group_match': sg_match
        }
    except Exception as e:
        return {'is_valid': False, 'error': str(e)}
```

## Metric Configuration for HELM

To implement these metrics in HELM:

1. **Create custom Metric class** (e.g., `CrystalStructureMatchMetric`)
   - Parse LLM-generated CIF text
   - Load reference CIF from Instance.references
   - Compute structural matching metrics
   - Return MetricResult with match_rate, avg_rmsd, validity_rate

2. **Create RunSpec with custom metrics**:
   ```python
   def get_material_generation_metric_specs():
       return [
           MetricSpec(
               class_name="helm.benchmark.metrics.crystal_structure_match_metric.CrystalStructureMatchMetric",
               args={},
           ),
       ]
   ```

## Notes

- **This is a complex evaluation task** requiring significant implementation work
- Consider starting with basic validity checking before implementing full structural matching
- May need to handle LLM outputs that don't strictly follow CIF format (parse robustly)
- Watch for computational cost - structural comparison can be expensive for large datasets

## References

- Materials Project: https://materialsproject.org/
- pymatgen documentation: https://pymatgen.org/
- MGB paper appendix (Section C) for detailed metric definitions
