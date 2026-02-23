# Fig-QA (Figurative Language Question Answering) - Annotator Notes

## Overview

Fig-QA tests language models' ability to interpret creative metaphors and figurative language through a Winograd schema-style task. Models must use commonsense reasoning to understand the implicit meanings conveyed through metaphorical expressions.

## Task Format

### Winograd Schema Structure

Each instance consists of:
1. **Startphrase**: A metaphorical statement (often a simile)
2. **Two endings**: Two possible interpretations
3. **Label**: Correct interpretation (0 or 1)

### Example 1:
```
Startphrase: "Her word had the strength of titanium."
Ending 1: Her promises can be believed.
Ending 2: Her promises cannot be trusted.
Correct: Ending 1 (titanium is strong → her word is strong/reliable)
```

### Example 2 (Paired/Contrast):
```
Startphrase: "Her word had the strength of a wine glass."
Ending 1: Her promises can be believed.
Ending 2: Her promises cannot be trusted.
Correct: Ending 2 (wine glass is fragile → her word is weak/unreliable)
```

## Metaphor Categories

### 1. Similes (Most Common)
- **Format**: "X is as Y as Z"
- **Example**: "The future is as bright as the sun" vs "The future is as bright as ink"
- **Reasoning**: Compare properties (sun is bright → future is good; ink is dark → future is bad)

### 2. Implicit Metaphors
- **Example**: "Sleeping over at his house is like spending a night at the Waldorf Astoria"
- **Reasoning**: Requires cultural knowledge (Waldorf Astoria = luxury hotel → very nice accommodations)

### 3. Property-Based Comparisons
- **Example**: "Her room was as messy as a housekeeper" vs "Her room was as messy as a tornado"
- **Reasoning**: Housekeeper implies cleanliness; tornado implies chaos

## Reasoning Requirements

### 1. Commonsense Knowledge
Models must know:
- Physical properties (titanium is strong, wine glass is fragile)
- Cultural references (Waldorf Astoria is luxurious)
- Typical associations (housekeeper → clean, tornado → messy)

### 2. Metaphorical Transfer
- Understand how properties transfer from source to target
- Example: titanium's strength → word's reliability
- Not just literal properties, but what they imply in context

### 3. Contrastive Reasoning
- Many examples are paired with opposite meanings
- Must distinguish subtle differences in vehicle terms
- Example: "bright as sun" vs "bright as ink"

## Dataset Structure

### Splits
- **Training**: 9,674 examples
- **Validation**: 1,094 examples (labels available)
- **Test**: 1,146 examples (labels hidden - use validation for dev)

### Fields
- `startphrase`: The metaphorical statement
- `ending1`: First interpretation option
- `ending2`: Second interpretation option
- `labels`: Correct answer (0 = ending1, 1 = ending2, -1 = hidden)
- `valid`: Validity flag (1 = valid example)

### Data Quality
- All metaphors are human-written
- Created through crowdsourcing with quality controls
- Paired examples ensure balanced difficulty

## Evaluation Methodology

### Primary Metric: Accuracy
- Percentage of correct interpretations
- Binary choice (A or B)
- Reported on validation or test split

### Evaluation Settings

**1. Zero-Shot**
- Prompt with instruction and question
- Model generates answer (A or B)
- Tests natural metaphor understanding

**2. Few-Shot**
- Provide 1-5 example metaphors with answers
- Model generalizes to new metaphors
- Tests learning from examples

**3. Fine-Tuning**
- Train on 9,674 training examples
- Evaluate on validation/test
- Tests trainable metaphor understanding

## Performance Benchmarks

From the paper (NAACL 2022):

### Human Performance
- **Accuracy**: ~95% (near ceiling)
- Humans find task intuitive and natural

### Model Performance (Zero/Few-Shot)
- **GPT-3 (175B)**: Significantly above chance but below human performance
- **GPT-2 variants**: Struggle with metaphorical reasoning
- **Gap**: Models fall 15-30% short of human performance in few-shot settings

### Fine-Tuned Performance
- Models can reach 70-85% accuracy with full training data
- Still below human performance
- Shows task difficulty even with supervision

## Key Challenges

### 1. Commonsense Reasoning
- Requires world knowledge beyond training data
- Must reason about properties and associations
- Not solvable through pattern matching alone

### 2. Nonliteral Language
- Cannot interpret metaphors literally
- Must understand figurative/implied meanings
- Requires pragmatic reasoning

### 3. Contrastive Examples
- Many paired examples with minimal changes
- Subtle differences lead to opposite meanings
- Tests fine-grained understanding

### 4. Cultural Knowledge
- Some metaphors require specific cultural context
- Geographic/temporal variations in metaphor understanding
- Tests breadth of commonsense knowledge

## Recommended HELM Evaluation Approach

### Primary Evaluation
1. **Use validation split** for development (labels available)
2. **Report accuracy** as primary metric
3. **Evaluate zero-shot and few-shot** (1, 3, 5 examples)

### Analysis Recommendations
1. **Error analysis by metaphor type** (similes vs implicit metaphors)
2. **Paired example analysis** (when models get one but not the paired opposite)
3. **Property reasoning failures** (which property transfers fail most)

### Prompting Strategy
```
Example prompt format:
Interpret this metaphor:
"Her word had the strength of titanium."

Which interpretation is correct?
A. Her promises can be believed.
B. Her promises cannot be trusted.

Answer:
```

## Comparison to Related Benchmarks

### WinoGrande
- **Similarity**: Winograd schema format
- **Difference**: WinoGrande tests pronoun resolution; Fig-QA tests metaphor interpretation

### CommonsenseQA
- **Similarity**: Tests commonsense reasoning
- **Difference**: CommonsenseQA is factual; Fig-QA tests creative/figurative language

### Metaphor Detection
- **Similarity**: Involves metaphorical language
- **Difference**: Detection identifies metaphors; Fig-QA interprets their meanings

## Citation

```bibtex
@inproceedings{liu-etal-2022-testing,
    title = "Testing the Ability of Language Models to Interpret Figurative Language",
    author = "Liu, Emmy  and
      Cui, Chen  and
      Zheng, Kenneth  and
      Neubig, Graham",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2204.12632"
}
```

## External Resources

- **GitHub**: https://github.com/nightingal3/Fig-QA
- **HuggingFace**: https://huggingface.co/datasets/nightingal3/fig-qa
- **Paper**: https://arxiv.org/abs/2204.12632
- **Leaderboard**: Available on Explainaboard
