# Evaluation Notes: MARS (Multimodal Analogical Reasoning dataSet)

**Paper**: Multimodal Analogical Reasoning over Knowledge Graphs (ICLR 2023)
**Source**: https://arxiv.org/abs/2210.00312
**Dataset**: https://github.com/zjunlp/MKG_Analogy

## Evaluation Type

MARS uses **open-ended generation with exact match** evaluation.

The task is formulated as analogical reasoning: (e_h, e_t) : (e_q : ?)

Given an example entity pair and a question entity, models must generate the correct answer entity.

## Metrics

### Primary Metric: Accuracy (Exact Match)
- Model output matches the correct entity ID (Q-code) or entity name
- Case-insensitive string matching

### Link Prediction Metrics (from paper)
The original paper evaluates this as a link prediction/ranking task:
- **Hits@1**: Percentage of correct answers ranked first
- **Hits@3**: Percentage of correct answers in top-3 predictions
- **Hits@10**: Percentage of correct answers in top-10 predictions
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of correct answers

## Dataset Statistics

- **Train**: 10,641 instances
- **Validation**: 1,020 instances
- **Test**: 1,362 instances

### Entity Statistics
- Total entities in MarKG: 11,292
- Analogical entities (with images): 2,063
- Relations: 192
- Knowledge triplets: 34,420
- Images: 76,424

### Test Set Statistics
- Unique answer entities: 262
- Total unique entities: 809

## Task Patterns

The paper describes two patterns:

1. **Single Analogical Reasoning** (mode 0)
   - All entities from same modality (all text OR all images)

2. **Blended Analogical Reasoning** (mode 1)
   - Entities from mixed modalities (some text, some images)

The scenario implementation treats both modes uniformly for evaluation.

## HELM Integration

### RunSpec Configuration

For open-ended generation with exact match:

```python
from helm.benchmark.run_specs import RunSpec, get_open_ended_generation_metric_specs

RunSpec(
    name="mars",
    scenario_spec=ScenarioSpec(
        class_name="scenarios.mars.scenario.MARSScenario",
        args={
            "use_images": False  # Set True if images are downloaded
        }
    ),
    metric_specs=get_open_ended_generation_metric_specs()
)
```

### With Images (Multimodal)

```python
RunSpec(
    name="mars:multimodal",
    scenario_spec=ScenarioSpec(
        class_name="scenarios.mars.scenario.MARSScenario",
        args={
            "use_images": True,
            "images_path": "/path/to/downloaded/MARS/images"
        }
    ),
    metric_specs=get_open_ended_generation_metric_specs()
)
```

## Expected Model Output Format

Models should output either:
- **Entity ID**: Q-code from Wikidata (e.g., "Q166231")
- **Entity Name**: Text description (e.g., "calcium deficiency")

The scenario provides both as correct references for flexible matching.

## Paper's Baseline Performance

The paper evaluates several baseline methods:

### Multimodal Knowledge Representation Methods
- **IKRL**: Knowledge representation learning with images
- **TransAE**: Translation-based autoencoder
- **RSME**: Relation-specific multimodal embeddings

### Transformer-based Methods
- **VisualBERT**: Vision-language pre-trained model
- **ViLBERT**: Vision-language BERT
- **ViLT**: Vision-and-Language Transformer
- **FLAVA**: Foundational Language And Vision Alignment
- **MKGformer**: Multimodal knowledge graph transformer

### MarT Framework (Paper's Proposed Method)
- **Pre-training**: On MarKG knowledge graph
- **Fine-tuning**: On MARS analogical reasoning task
- Achieved best performance on the benchmark

## Image Data Download

**IMPORTANT**: Images are NOT included in the GitHub repository due to size (76,424 images).

Download from:
- **Google Drive**: https://drive.google.com/file/d/1AqnyrA05vKngfEbhw1mxY5qEoaqiKsC1/view
- **Baidu Pan**: https://pan.baidu.com/s/1WZvpnTe8m0m-976xRrH90g (code: 7hoc)

Extract to: `MarT/dataset/MARS/images/`

Images are organized by entity ID:
```
images/
  Q1501/
    02009000.jpg
    02009001.jpg
    ...
  Q6574/
    image1.jpg
    ...
```

## Text-Only Mode

The scenario can run in text-only mode (default) using entity descriptions from:
- `MarKG/entity2text.txt` - Short descriptions
- `MarKG/entity2textlong.txt` - Long descriptions (optional)

This allows evaluation without downloading the large image dataset.

## Evaluation Considerations

### Multimodal vs. Text-Only
- **With images**: Tests true multimodal reasoning capabilities
- **Text-only**: Tests analogical reasoning from textual descriptions
- Performance may differ significantly between modes

### Entity Ambiguity
- Multiple entities may have similar descriptions
- Visual information can disambiguate entities
- Text-only evaluation may be more challenging

### Knowledge Graph Context
- Original paper uses MarKG background knowledge for pre-training
- HELM evaluation tests models without explicit KG access
- Models rely on internal knowledge or entity descriptions

## Related Work

This benchmark is based on classical analogical reasoning theory:
- **Structure-Mapping Theory** (Gentner, 1983)
- **Abduction-Mapping-Induction** pipeline
- Extended to multimodal knowledge graphs

## Notes

- Dataset uses Wikidata entity IDs (Q-codes) and property IDs (P-codes)
- Relations describe the analogical relationship (e.g., P828 = "has cause")
- Task requires both visual understanding and relational reasoning
- Analogical reasoning is a fundamental creative thinking capability
