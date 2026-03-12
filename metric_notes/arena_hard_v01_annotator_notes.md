# Annotator Requirements: Arena-Hard v0.1

Source: https://github.com/lmarena/arena-hard-auto
Paper: Li et al., 2024 (Arena-Hard-Auto)
Dataset: lmarena-ai/arena-hard-auto-v0.1

## Configuration for LLMAsJuryAnnotator

**Judge model**: GPT-4-turbo (or Gemini-1.5-Pro)
**Task**: Pairwise comparison - model vs. baseline
**Baseline**: GPT-4-turbo responses
**Metric**: Win Rate (% of wins against baseline)

## Judge Prompt Template

```
Please act as an impartial judge and evaluate the quality of the responses provided
by two AI assistants to the user question displayed below. You should choose the
assistant that follows the user's instructions and answers the user's question better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]

[System]
Please identify which assistant provides a better response. Output your verdict as
"Assistant A" or "Assistant B".
```

## Metrics

**Win Rate**:
- Definition: Percentage of instances where model beats baseline
- Calculation: `wins / total_comparisons`
- Baseline: GPT-4-turbo reference answers

**Tie-Adjusted Win Rate**:
- Some implementations count ties as 0.5 wins
- `(wins + 0.5 * ties) / total_comparisons`

## Dataset Structure (v0.1)

- **Total prompts**: 500
- **Clusters**: 250 topic categories
- **Format**: Multi-turn conversations (typically single-turn used)
- **Difficulty**: Hard, challenging queries from real users

**Note**: DARLING paper references v2.0 with 750 prompts (500 general + 250 creative writing).
This implementation uses v0.1 (500 prompts) which is publicly available.

## Sample Clusters

- "ABC Sequence Puzzles & Groups"
- "Delete System32 with Rust"
- "Philosophy & Theology Reviews"
- "Text-Based RPG Creation"
- "Audio Signal Direction Detection"
- "HDL Design and Verification"
- And 244 more diverse topics

## Judge Models

Arena-Hard supports multiple judges:
- **GPT-4-turbo** (recommended)
- **GPT-4-1106-preview**
- **Gemini-1.5-Pro** (alternative)
- **Claude-3-Opus** (alternative)

Different judges may produce slightly different win rates.

## Notes for HELM Adaptation

- **Pairwise comparison**: Model vs. GPT-4-turbo baseline
- **Single-turn**: Use `turns[0]['content']` as prompt
- **Multi-turn support**: Future extension could use full conversation history
- **Empty references**: No ground-truth answers (pure generation + judge)
- **Win rate metric**: Requires custom implementation

## Limitations

- v0.1 doesn't explicitly separate creative writing from other tasks
- All 500 prompts categorized uniformly as "arena-hard-v0.1"
- For creative-writing-specific evaluation, see EQBench Creative Writing v3
- v2.0 with 250 creative writing subset may become available later

## References

- GitHub: https://github.com/lmarena/arena-hard-auto
- HuggingFace v0.1: https://huggingface.co/datasets/lmarena-ai/arena-hard-auto-v0.1
- Chatbot Arena: https://lmarena.ai/
- DARLING paper: arXiv:2509.02534 (references v2.0)
