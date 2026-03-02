# InfoChartQA Scenario

## Overview

InfoChartQA is a multimodal question answering benchmark that evaluates models' ability to understand infographic charts enriched with creative visual elements like pictograms, icons, and visual metaphors.

## Dataset

- **Source**: HuggingFace ([Jietson/InfoChartQA](https://huggingface.co/datasets/Jietson/InfoChartQA))
- **GitHub**: https://github.com/CoolDawnAnt/InfoChartQA
- **Total Size**: 58,857 questions across 5,948 chart pairs
- **Modality**: Vision-Language (chart images + text questions)

## Splits

| Split | Size | Description |
|-------|------|-------------|
| `text` | 50,920 | Text-based QA on charts |
| `visual_metaphor` | 462 | Visual metaphor understanding |
| `visual_basic` | 7,475 | Basic visual element understanding |

## Usage

```python
from scenarios.infochartqa.scenario import InfoChartQAScenario

# Load all splits
scenario = InfoChartQAScenario(subset="all")

# Or load specific split
scenario_vm = InfoChartQAScenario(subset="visual_metaphor")
scenario_text = InfoChartQAScenario(subset="text")
scenario_visual = InfoChartQAScenario(subset="visual_basic")
```

## Task Types

### Question Types (26 categories)
- **Value extraction**: Single element, element at time
- **Difference**: Numerical comparison, yes/no comparison
- **Trend analysis**: Temporal trend description
- **Categorization**: Target filtering, grouping, category identification
- **Aggregation**: Sum, average, median, count
- **Ranking**: Find extremes, rank by value
- **Correlation**: Association analysis

### Chart Types (54 types)
- Simple: bar, line, pie, donut, scatter
- Grouped: grouped bar/line/area
- Stacked: stacked bar/area
- Advanced: treemap, funnel, radar, sankey, heatmap
- Maps: value-based map

## Evaluation

- **Metric**: Exact match accuracy
- **Format**: Answers must match ground truth exactly
- **Instructions**: Task-specific formatting requirements provided in dataset

## Special Features

### Instructions Field
Questions include optional instructions that specify:
- Output format constraints
- Calculation methods
- Value formatting requirements

These must be concatenated with the question text.

### Visual Metaphor Split
462 questions specifically testing understanding of visual metaphors and creative visual design elements in charts.

### Extra Figures (visual_basic)
Some questions include bounding boxes (`extra_input_figure_bboxes`) indicating cropped sections to focus on specific visual elements. Format: `[x, y, width, height]`.

**Note**: Current implementation includes full chart image. For models requiring cropped sections, images would need to be pre-processed.

## Difficulty Levels

- **Easy**: Basic chart reading
- **Moderate**: Multi-step reasoning
- **Hard**: Complex visual metaphor understanding

## Implementation Notes

- Images are stored as URLs (remote hosting)
- Uses HELM's `MediaObject` for multimodal input
- `MultimediaObject` combines image and text
- References tagged with `CORRECT_TAG` for evaluation

## Citation

```bibtex
@misc{infochartqa,
  title={InfoChartQA: A Dataset for Visual Question Answering on Infographic Charts},
  author={InfoChartQA Team},
  year={2024},
  url={https://github.com/CoolDawnAnt/InfoChartQA}
}
```
