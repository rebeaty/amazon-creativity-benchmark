# Material Generation Benchmark (MGB)

**Paper:** MGB: The Material Generation Benchmark
**Source:** [OpenReview](https://openreview.net/forum?id=K15Dqxm0ge)
**Code:** [CDVAE GitHub](https://github.com/txie-93/cdvae)

## Overview

The Material Generation Benchmark evaluates LLMs on generating crystal structures in CIF (Crystallographic Information File) format from material specifications. This is a text-to-structure generation task where models receive material composition and properties as input and must produce valid, physically plausible crystal structures.

## Datasets Onboarded

### 1. MP-20 (`mp20_scenario.py`)
- **Size:** 45,231 stable materials from Materials Project/ICSD
- **Atoms:** 1-20 atoms per unit cell
- **Elements:** 89 elements
- **Test set:** 9,046 instances
- **Input properties:** Formula, space group, formation energy, band gap, energy above hull

### 2. PEROV-5 (`perov5_scenario.py`)
- **Size:** 18,928 perovskite materials (ABX3 formula)
- **Atoms:** 5 atoms per unit cell (fixed)
- **Elements:** 56 elements
- **Test set:** 3,785 instances
- **Input properties:** Formula, heat of formation (all/reference), direct/indirect band gap
- **Source:** Water-splitting database

### 3. Carbon-24 (`carbon24_scenario.py`)
- **Size:** 10,153 carbon allotrope structures
- **Atoms:** 6-24 atoms per unit cell
- **Elements:** 1 element (carbon only)
- **Test set:** 2,030 instances
- **Input properties:** Material ID, energy per atom, pressure (10 GPa)
- **Source:** AIRSS (ab initio random structure searching)

### 4. MOF (Metal-Organic Frameworks) (`mof_scenario.py`)
- **Size:** 324,426 MOF structures from Boyd et al. (2019)
- **Atoms:** Hundreds of atoms per structure (complex porous materials)
- **Components:** Metal nodes + organic linkers
- **Test set:** ~32,443 instances (10% split, 8:1:1 train/valid/test)
- **Input properties:** Metal nodes, organic linkers, topology
- **Source:** Zenodo (15187230), MOFFlow (ICLR 2025)
- **Benchmark:** MOFFlow baseline achieves 31.69% match rate

## Task Format

**Input (Prompt):**
```
Generate a crystal structure in CIF format for the following material:

Formula: {formula}
Space group: {spacegroup}
Formation energy per atom: {formation_energy} eV/atom
Band gap: {band_gap} eV
Energy above hull: {e_above_hull} eV/atom

Output the structure in standard CIF format:
```

**Expected Output:** CIF format crystal structure (structured text, 500-2000 characters)

**Example CIF Output:**
```
# generated using pymatgen
data_GaTe
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   4.13459945
_cell_length_b   4.13459945
_cell_length_c   18.42557000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   119.99999121
...
```

## Evaluation

This benchmark requires **custom evaluation metrics** that are NOT available in standard HELM. See `metric_notes.md` for detailed implementation requirements.

### Required Metrics

1. **Structural Matching**
   - Match rate (% of generated structures matching references)
   - RMSD (root mean square deviation of atomic positions)

2. **Validity**
   - Can parse as valid CIF file
   - Chemical composition matches specification
   - No atomic collisions

3. **Physical Plausibility**
   - Reasonable density
   - Proper symmetry
   - Valid interatomic distances

### Implementation Status

- ✅ **Scenarios created** - Data loading and prompt formatting
- ⚠️ **Metrics pending** - Requires pymatgen/ASE integration for structural comparison
- ⚠️ **RunSpec pending** - Need to create RunSpec configuration

## Data Access

All datasets are publicly available via the CDVAE repository:
- MP-20: `https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/{split}.csv`
- PEROV-5: `https://raw.githubusercontent.com/txie-93/cdvae/main/data/perov_5/{split}.csv`
- Carbon-24: `https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/{split}.csv`

Format: CSV files with columns for material properties and CIF structure

## Notes

- **Creativity Aspect:** While this evaluates technical correctness (structure validity), the *de novo* generation of novel stable materials has creative elements (exploring chemical space, discovering new compounds)
- **Domain-Specific:** Requires materials science expertise and tools for proper evaluation
- **Text-to-Structure:** Treats CIF as structured text output, bridging text generation and scientific computing
- **Evaluation Complexity:** Cannot use standard text metrics (BLEU, ROUGE); needs structural comparison algorithms

## References

- Paper: MGB: The Material Generation Benchmark (OpenReview 2025)
- CDVAE: Crystal Diffusion Variational Autoencoder (ICLR 2022)
- Materials Project: https://materialsproject.org/
- pymatgen: https://pymatgen.org/ (Python library for materials analysis)
