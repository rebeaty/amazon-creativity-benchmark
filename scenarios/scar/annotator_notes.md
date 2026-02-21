# Evaluation Notes: SCAR (Scientific Analogical Reasoning with Structure Abduction)

**Paper**: Beneath Surface Similarity: Large Language Models Make Reasonable Scientific Analogies after Structure Abduction (EMNLP 2023 Findings)
**Source**: https://aclanthology.org/2023.findings-emnlp.160
**Dataset**: https://github.com/siyuyuan/scar

## Evaluation Type

SCAR uses **open-ended generation with structured output matching** evaluation.

The task is formulated as analogical structure abduction: given two systems with background information, identify the term mappings that form analogies between elements in the two systems.

## Metrics

### Primary Metric: Exact Match (Set-based)

- Model output is parsed as a list of term pairs (mappings)
- Each mapping is a two-element list: [term_from_system_a, term_from_system_b]
- Evaluation compares the set of generated mappings against ground truth mappings
- Scoring options:
  - **Strict match**: All mappings must be correct (set equality)
  - **Partial match**: Precision/Recall/F1 on individual mappings
  - **Relaxed match**: Allow for case-insensitive and punctuation-normalized matching

### Additional Metrics (from paper)

The original paper evaluates using:
- **Accuracy**: Percentage of instances where all mappings are correctly identified
- **Precision**: Correct mappings / Generated mappings
- **Recall**: Correct mappings / Gold mappings
- **F1-score**: Harmonic mean of precision and recall

## Dataset Statistics

- **Total instances**: 400 scientific analogies
- **Domains**: 13 fields (Biology, Physics, Chemistry, Computer Science, Mathematics, Engineering, Geography, History, Literature, Philosophy, Economics, Art, Sports)
- **Mappings per instance**: 2-14 (average: 4.0)
- **Split**: All 400 instances used as test set (no official train/val splits)

### Domain Statistics

Examples of cross-domain analogies:
- Biology ↔ Engineering (e.g., cell ↔ factory)
- Physics ↔ Computer Science
- Chemistry ↔ Economics
- Mathematics ↔ Philosophy

## Task Format

### Input Format

Each instance provides:
1. **Scenario 1** (System A): Name, domain, background description, list of items
2. **Scenario 2** (System B): Name, domain, background description, list of items

Example:
```
Scenario 1: Solar System
Domain: Physics
Background: [detailed description]
Items in Scenario 1: Newton, Sun, Earth

Scenario 2: Atom Structure
Domain: Physics
Background: [detailed description]
Items in Scenario 2: Nucleus, Faraday, Electron
```

### Expected Output Format

Models should output mappings in the format:
```
[['Newton','Faraday'], ['Sun','Nucleus'], ['Earth','Electron']]
```

- List of lists format
- Each mapping is a two-element list
- Order of mappings within the list doesn't matter (set comparison)
- Order within each pair matters (first element from System A, second from System B)

## HELM Integration

### RunSpec Configuration

For open-ended generation evaluation:

```python
from helm.benchmark.run_specs import RunSpec, get_open_ended_generation_metric_specs

RunSpec(
    name="scar",
    scenario_spec=ScenarioSpec(
        class_name="scenarios.scar.scenario.SCARScenario",
        args={}
    ),
    metric_specs=get_open_ended_generation_metric_specs()
)
```

### Custom Evaluation

For accurate evaluation of structured output, consider implementing a custom metric:

```python
from helm.benchmark.metrics.metric import Metric

class SCARMappingMetric(Metric):
    """
    Evaluates SCAR mappings using set-based comparison.

    Parses model output as list of pairs and compares against gold mappings.
    """

    def evaluate(self, predicted_text: str, reference_texts: List[str]):
        # Parse predicted_text as list of pairs
        # Compare against reference (gold mappings)
        # Return precision, recall, F1, exact_match
```

## Paper's Baseline Performance

The paper evaluates several LLMs on SCAR:

### Models Tested
- **GPT-3** (text-davinci-003)
- **ChatGPT** (gpt-3.5-turbo)
- **GPT-4**
- **LLaMA-based models**

### Key Findings (from paper)
- State-of-the-art LLMs struggle with structure abduction
- GPT-4 achieves best performance but still faces challenges
- Cross-domain transfer is harder than within-domain analogies
- Providing explicit background information significantly improves performance
- Models often focus on surface similarity rather than relational structure

### Domain Transfer Analysis
- Analogies between similar domains (e.g., Physics ↔ Engineering) show higher accuracy
- More disparate domains (e.g., Art ↔ Computer Science) show lower accuracy
- This suggests models rely partially on domain knowledge overlap

## Evaluation Considerations

### Output Parsing Challenges
- Models may generate mappings in various formats (not always strict Python list format)
- May include explanations or reasoning before/after the mapping list
- Need robust parsing to extract the mapping pairs from free-form text
- Consider regex patterns to extract pairs in various formats

### Partial Credit
- Some analogies have multiple valid mappings
- Consider giving partial credit for correct subset of mappings
- F1-score is better metric than strict accuracy for this reason

### Human Evaluation
- The paper includes human evaluation showing that model-generated analogies can be reasonable even when not matching gold standard
- Consider qualitative analysis of "reasonable but different" mappings

## Related Tasks

This benchmark is related to:
- **Analogical reasoning**: Classic A:B::C:? format (e.g., king:queen::man:?)
- **Knowledge graph completion**: Predicting relations between entities
- **Structure mapping**: Cognitive science theory of analogy (Gentner, 1983)

Key difference: SCAR requires identifying multiple mappings simultaneously and reasoning about complex systems, not just single term pairs.

## Alternative Name

This benchmark is also referred to as the **"Relational Structure Identification (RSI) Test"** in some references and in the benchmarks.json tracking file.

## Notes

- All instances are in English (Chinese version also available in repository)
- Background descriptions are lengthy (100-300 words per system)
- Some systems appear multiple times paired with different systems
- Dataset was created through annotation with double-check quality control
- Annotators were compensated above local minimum wage
- Creative aspect: Requires identifying novel structural similarities across disparate domains
