# Metric Notes: Speak-to-Structure (S2-Bench / TOMG-Bench)

Source: arXiv:2412.14642; https://phenixace.github.io/tomgbench/
Data:   https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench

## Original Evaluation Metrics (from paper, Section 4)

The paper uses three evaluation metrics applied per subtask:

| Metric | Description | Applies To |
|--------|-------------|------------|
| **Validity** | % of outputs that are valid SMILES (parseable by RDKit) | All |
| **Success Rate** | % of valid outputs that satisfy the stated constraint | All |
| **Novelty** | % of valid molecules not in training set (ZINC/ChEMBL) | MolCustom |
| **Similarity** | Tanimoto similarity of edited molecule to source | MolEdit, MolOpt |

## Tier 1: HELM-Computable (No RDKit)

### SMILES Syntax Validity (regex-based approximation)

A SMILES string uses atoms (C, N, O, S, F, Cl, Br, I, P, B, Si),
brackets `[]`, bonds `-=#:`, rings `1-9%xx`, and branches `()`.

```python
import re

def is_plausible_smiles(text: str) -> bool:
    """Basic syntactic check — not chemically validated."""
    s = text.strip()
    if not s or len(s) < 2:
        return False
    # Must contain at least one atom letter
    if not re.search(r'[CNOSPBFIcnos]', s):
        return False
    # Must not contain spaces or forbidden characters
    if re.search(r'[\s]', s):
        return False
    # Balanced parentheses and brackets
    return s.count('(') == s.count(')') and s.count('[') == s.count(']')
```

### Length Heuristic

Valid drug-like SMILES are typically 5–500 characters.
Outputs outside this range are almost certainly invalid.

## Tier 2: RDKit-Based Evaluation

Install: `pip install rdkit`

### SMILES Validity (authoritative)

```python
from rdkit import Chem

def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None
```

### MolCustom — Constraint Satisfaction

**AtomNum**: Check atom counts match the instruction.
```python
def check_atom_num(smiles: str, target_counts: dict) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    actual = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        actual[sym] = actual.get(sym, 0) + 1
    for element, count in target_counts.items():
        if actual.get(element, 0) != count:
            return False
    return True
```

**FunctionalGroup**: Use RDKit SMARTS patterns.
```python
FUNCTIONAL_GROUP_SMARTS = {
    "hydroxyl":   "[OX2H]",
    "aldehyde":   "[CX3H1](=O)[#6]",
    "ketone":     "[CX3](=O)([#6])[#6]",
    "carboxyl":   "[CX3](=O)[OX2H1]",
    "amine":      "[NX3;H2,H1;!$(NC=O)]",
    "benzene rings": "c1ccccc1",
    # ... (full mapping in paper Appendix)
}
```

**BondNum**: Count bond types using RDKit bond enumeration.

**BasicProp**: Requires property prediction models (logP, boiling point, etc.).

### MolEdit — Tanimoto Similarity

```python
from rdkit.Chem import AllChem, DataStructs

def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)
```

Success = Tanimoto ≥ 0.4 AND the specified group is present/absent.

### MolOpt — Property Improvement

```python
from rdkit.Chem import Descriptors

def compute_logp(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolLogP(mol) if mol else None

def compute_qed(smiles: str) -> float:
    from rdkit.Chem import QED
    mol = Chem.MolFromSmiles(smiles)
    return QED.qed(mol) if mol else None

def compute_mr(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolMR(mol) if mol else None
```

Success = property improved in the stated direction vs. source molecule.

## Baseline Results (from paper, Table 2)

| Model | Validity | Success Rate (avg) |
|-------|----------|--------------------|
| GPT-4o | ~85% | ~35% |
| Claude-3.5-Sonnet | ~83% | ~32% |
| Llama-3.1-8B (fine-tuned on OpenMolIns) | ~92% | ~58% |
| Random baseline | ~45% | ~8% |

Key finding: Fine-tuned small models (Llama-3.1-8B with OpenMolIns) outperform
GPT-4o and Claude-3.5-Sonnet, suggesting general LLMs lack molecule-specific
generation capabilities without domain fine-tuning.

## Data Loading Note

`load_dataset("Duke-de-Artois/TOMG-Bench")` fails with:
  `ValueError: Couldn't infer the same data file format for all splits.
   Got {train: csv, validation: text, test: csv}`

The scenario uses per-subtask CSV download via `urllib.request.urlretrieve`.
Each CSV URL follows:
  https://huggingface.co/datasets/Duke-de-Artois/TOMG-Bench/resolve/main/
  benchmarks/open_generation/{Task}/{Subtask}/test.csv

Cached locally in `output_path` after first download.

## Subtask Reference

| Task | Subtask | Input Fields | Evaluation |
|------|---------|-------------|------------|
| MolCustom | AtomNum | Instruction | Atom count check (RDKit) |
| MolCustom | BasicProp | Instruction | Property prediction (external model) |
| MolCustom | BondNum | Instruction | Bond count check (RDKit) |
| MolCustom | FunctionalGroup | Instruction | SMARTS matching (RDKit) |
| MolEdit | AddComponent | Instruction, molecule | Tanimoto + SMARTS check |
| MolEdit | DelComponent | Instruction, molecule | Tanimoto + SMARTS check |
| MolEdit | SubComponent | Instruction, molecule | Tanimoto + SMARTS check |
| MolOpt | LogP | Instruction, molecule, logP | LogP comparison (RDKit) |
| MolOpt | MR | Instruction, molecule, MR | MR comparison (RDKit) |
| MolOpt | QED | Instruction, molecule, QED | QED comparison (RDKit) |
