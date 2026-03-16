# Annotator Requirements: AlpacaEval 2.0

Source: https://github.com/tatsu-lab/alpaca_eval
Paper: "Length-Controlled AlpacaEval" (Dubois et al., 2024) - arXiv:2404.04475

## Configuration for LLMAsJuryAnnotator

**Judge model**: GPT-4-turbo (or compatible LLM)
**Task**: Pairwise comparison - rank two model outputs
**Dimensions**: Overall quality, helpfulness, preference
**Scale**: Binary win/loss decision

## Judge Prompt Template

```
I want you to create a leaderboard of different large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans.

Here is the prompt:
{
    "instruction": "{instruction}"
}

Here are the outputs of the models:
[
    {
        "model": "model_1",
        "answer": "{output_1}"
    },
    {
        "model": "model_2",
        "answer": "{output_2}"
    }
]

Please rank the models based on the quality of their answers.
```

The judge outputs:
```json
{
    "model_1": 1,  // or 2 if model_2 is better
    "model_2": 2   // or 1 if model_1 is better
}
```

## Metrics

### 1. Win Rate (WR)
- **Definition**: Percentage of instances where model output beats baseline
- **Calculation**: `wins / total_comparisons`
- **Baseline**: GPT-4-turbo responses from `output` field

### 2. Length-Controlled Win Rate (LCWR)
- **Definition**: Win rate adjusted for length bias
- **Method**: Fit generalized linear model to predict preferences based on length differences
- **Purpose**: Addresses judges' tendency to favor longer outputs
- **Calculation**: Answers counterfactual "What would preference be if outputs had same length?"

## Length Bias Correction

AlpacaEval judges show strong length bias:
- Spearman correlation with Chatbot Arena: 0.94 (raw WR)
- Spearman correlation with Chatbot Arena: 0.98 (LCWR)

**Implementation**:
1. Compute length difference: `len(output_1) - len(baseline)`
2. Fit GLM: `preference ~ length_diff + other_features`
3. Predict preference at `length_diff = 0`

## Notes for HELM Adaptation

- **Pairwise setup**: Each comparison is model vs. baseline (not model vs. model)
- **Baseline**: Use `output` field from dataset as reference (text_davinci_003 or GPT-4)
- **Judge prompt**: Template available in `src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/alpaca_eval.txt`
- **Multiple annotators**: Can use different judge models (GPT-4, Claude, etc.) for comparison
- **Length tracking**: Record output lengths for LCWR calculation

## Dataset Statistics

- **Total prompts**: 805
- **Source datasets**: helpful_base, koala, self_instruct, vicuna, etc.
- **Diversity**: Instruction-following, creative writing, reasoning, QA
- **Format**: Plain instructions (no few-shot examples)

## References

- Paper: https://arxiv.org/abs/2404.04475
- Code: https://github.com/tatsu-lab/alpaca_eval
- Leaderboard: https://tatsu-lab.github.io/alpaca_eval/
- DARLING usage: arXiv:2509.02534 (creative writing evaluation)
