# Project Context for Claude

## Benchmark Onboarding Learnings

Add issues and patterns here as you discover them. Everyone on the team benefits.

### Dataset Quirks

| Benchmark | Issue | Solution |
|-----------|-------|----------|
| RiddleSense | Test split has no labels (empty `answerKey`) | Use validation split instead |
| HumorDB | Multimodal only (images required) | Skip - not suitable for text-to-text HELM |
| ANALOBENCH | Field is `Sentence` not `Story` | Check actual dataset keys before coding |

### Common Patterns

- **Datasets requiring `trust_remote_code`**: Add `trust_remote_code=True` to `load_dataset()`
- **Very long text fields (>500 chars)**: Likely model outputs, not prompts - skip these
- **Fields with "gpt", "claude", "response", "output" in name**: Model outputs, skip

### Red Flags to Skip

- Benchmark is image/video only (no text component)
- Dataset only contains model-generated responses
- No evaluation framework exists
- Benchmark name includes "safety", "toxicity", "bias"

### Style Conventions

- Scenario class names: `{Benchmark}Scenario` (e.g., `RiddlesenseScenario`)
- `name` field: lowercase with underscores (e.g., `riddlesense`)
- `description` field: data source reference, not task description
- Always include `tags = ["creativity", ...]`
