# Evaluation Metrics: Materials Transformers

Source: "Materials Transformers Language Models for Generative Materials Design" (Fu et al., 2022)
Paper: https://arxiv.org/abs/2206.13578
Code: https://github.com/usccolumbia/mtransformer

## Task Overview

Models generate chemical formulas in **expanded element form** (space-separated element sequences ending with period):
```
Input:  "Generate a chemically valid material composition:"
Output: "Li Li O ."  (represents Li₂O)
```

This is an **open-ended unconditional generation task**. Models learn the distribution of valid material compositions and generate novel formulas.

## Required Custom Metrics

Standard HELM metrics (BLEU, ROUGE, exact match) are **not applicable**. This benchmark requires specialized materials science validation metrics.

### 1. **Validity Metrics**

Evaluate whether generated formulas are chemically valid:

#### a) Charge Neutrality (CN)
- **Definition**: Sum of oxidation states equals zero
- **Calculation**: Use pymatgen `Composition.oxi_state_guesses()` to check charge balance
- **Baseline**: Random generation achieves 10.13% CN
- **Target**: MT models achieve 97.54% CN

#### b) Electronegativity Balance (EB)
- **Definition**: Pauling electronegativity differences follow chemical bonding rules
- **Calculation**: Check EN differences between elements using pymatgen
- **Baseline**: Random generation achieves 5.91% CN+EB
- **Target**: MT models achieve 91.40% CN+EB

**Implementation**:
```python
from pymatgen.core import Composition

def is_charge_neutral(formula_str):
    """Check if formula is charge neutral"""
    try:
        comp = Composition(formula_str)
        # Try to guess oxidation states
        oxi_guesses = comp.oxi_state_guesses()
        return len(oxi_guesses) > 0
    except:
        return False
```

### 2. **Uniqueness**

- **Definition**: Percentage of unique formulas among all generated samples
- **Calculation**: `unique_formulas / total_generated`
- **Purpose**: Measures diversity of generation (avoid mode collapse)

### 3. **Recovery Rate**

- **Definition**: Percentage of held-out test formulas that can be regenerated
- **Method**: Generate N formulas, check how many match test set
- **Purpose**: Tests model's coverage of the materials space

### 4. **Novelty**

- **Definition**: Percentage of valid formulas NOT in the training set
- **Calculation**:
  ```
  novelty = (valid_generated - seen_in_training) / valid_generated
  ```
- **Purpose**: Measures creative discovery of new materials

## Evaluation Pipeline

1. **Generate**: Model produces N formulas (e.g., N=1000 per test instance)
2. **Parse**: Convert expanded form to chemical formula
   - "Li Li O ." → "Li₂O"
3. **Validate**: Check CN and EB using pymatgen
4. **Deduplicate**: Count unique formulas
5. **Check Novelty**: Compare against training set
6. **Compute Recovery**: Check overlap with test set

## Metrics Summary Table

| Metric | Formula | Baseline | MT Models |
|--------|---------|----------|-----------|
| Charge Neutrality | CN formulas / total | 10.13% | **97.54%** |
| CN + EB | (CN ∩ EB) / total | 5.91% | **91.40%** |
| Uniqueness | unique / total | - | High (model-dependent) |
| Recovery Rate | recovered / test_set_size | - | Tested via leave-out |
| Novelty | novel / valid | - | High (model-dependent) |

## Implementation Requirements

### Dependencies
- **pymatgen**: Chemical composition validation, oxidation state checking
- **pandas**: Data handling for formula comparison

### Additional Data Needed
- **Training set formulas**: For novelty calculation (included in MT_dataset)
- **Element properties**: Electronegativity values (from pymatgen)

## Notes for HELM Integration

- **RunSpec**: Requires custom metric implementation (not in standard HELM)
- **Generation**: Models should generate multiple samples per prompt (N=100-1000)
- **Comparison**: Cannot use reference-based metrics; evaluation is property-based
- **Baseline**: Include random generation baseline for context (pseudo-random element sampling)

## Recommended Evaluation Protocol

1. Generate 1000 formulas per model
2. Compute all 4 metrics (CN, CN+EB, uniqueness, novelty)
3. Report mean ± std across multiple generation runs
4. Compare against baseline and original paper results

## Future Extensions

- **Property prediction**: Validate bandgap for Bandgap-30K dataset
- **Structure generation**: Extend to 3D structures (complementary to MGB benchmark)
- **Diversity metrics**: FCD (Fréchet ChemNet Distance) for chemical space coverage
