# Annotator Requirements: Open-domain Dialogue Generation for Diversity

Source: ArXiv 2412.03343 - Improving Linguistic Diversity of Large Language Models with Possibility Exploration Fine-Tuning
Repository: https://github.com/mailong25/peft_diversity

## Task Overview

**Goal**: Generate multiple semantically diverse dialogue responses to the same conversation context.

**Challenge**: LLMs typically produce predictable, conservative responses when given the same prompt. This benchmark evaluates the ability to generate diverse yet coherent responses by varying a "possibility number" parameter.

## Task Format

### Input
- **Conversation Context**: Multi-turn dialogue history with alternating speakers (Person A and Person B)
- **Possibility Number**: An integer k (typically 1-5 or 1-10) that guides the model toward different response variations

### Output
- **Response**: A short dialogue response (≤ 25 words) from Person B's perspective
- **Requirement**: Responses for different possibility numbers should be semantically distinct from each other

### Example

**Context:**
```
Person A: I have been working in retail while I finish up school, same as you, I suppose.
Person B: What are you studying?
```

**Expected diverse responses:**
- Possibility #1: I'm wrapping up my psychology and human resources studies.
- Possibility #2: I'm into data and technology, so I'm majoring in computer science.
- Possibility #3: I'm studying business management and marketing.
- Possibility #4: Biology and environmental science are my focus areas.
- Possibility #5: I'm pursuing a degree in education and child development.

## Evaluation Methodology

The benchmark uses **multiple complementary metrics** to assess both diversity and quality:

### 1. Semantic Diversity (Primary Metric)

**Approach**: Measure pairwise cosine similarity between sentence embeddings

**Implementation**:
- Encode all responses for the same context using a sentence embedding model
- Calculate pairwise cosine similarities
- Average similarity = semantic diversity score (lower is better)

**Formula**:
```
semantic_diversity = 1 - avg(cosine_similarity(emb_i, emb_j)) for all pairs i,j
```

**Interpretation**:
- High similarity (>0.8): Responses are too similar, not diverse
- Medium similarity (0.5-0.8): Moderate diversity
- Low similarity (<0.5): High diversity

### 2. N-gram Diversity (Lexical Metric)

**Distinct-1 (Unigram Diversity)**:
```
Distinct-1 = |unique_unigrams| / |total_unigrams|
```

**Distinct-2 (Bigram Diversity)**:
```
Distinct-2 = |unique_bigrams| / |total_bigrams|
```

**Calculation**: Across all responses generated for all test contexts

**Interpretation**:
- Higher Distinct-N = more lexical diversity
- Range: 0.0 to 1.0
- Typical values: 0.3-0.7 for Distinct-1, 0.5-0.9 for Distinct-2

### 3. Coherence Evaluation (Quality Metric)

**Method**: LLM-as-Judge

**Judge Models**:
- Primary: Llama-based chat model (via Together AI)
- Fallback: GPT-4 (for low-scoring responses)

**Coherence Prompt Template**:
```
Given this dialog:

[formatted conversation context]

Does this next response from Person B make coherent sense?
"Person B: [generated_response]"

Begin your evaluation by providing a short assessment. Then, rate the coherence of
Person B's response on a scale from 1 to 10 by strictly following this example format:
'Coherence rating: [5]'

Coherence assessment:
```

**Scale**: 1-10
- 1-3: Incoherent, nonsensical
- 4-5: Somewhat incoherent, poor fit
- 6-7: Coherent but may lack relevance
- 8-9: Coherent and relevant
- 10: Perfectly coherent and highly relevant

**Incoherence Rate**: Percentage of responses with coherence ≤ 5
- Lower incoherence rate = better quality
- Target: <10% incoherence rate

### 4. Diversity-Coherence Trade-off

**Key Challenge**: Maximizing diversity while maintaining coherence

**Ideal Performance**:
- High semantic diversity (low avg similarity)
- High n-gram diversity (high Distinct-1/2)
- High coherence scores (≥8)
- Low incoherence rate (<10%)

**Common Trade-offs**:
- Random/diverse beam search: High diversity, low coherence
- Greedy/beam search: High coherence, low diversity
- Temperature sampling: Variable, depends on temperature

## Evaluation Protocol

### Step 1: Generate Multiple Responses

For each test context, generate N responses (typically 5-10) by varying the possibility number:
- Response 1: possibility #1
- Response 2: possibility #2
- ...
- Response N: possibility #N

### Step 2: Calculate Diversity Metrics

**Per-Context Diversity**:
- Calculate semantic similarity between all pairs of N responses
- Average pairwise similarity → semantic diversity score

**Global Diversity**:
- Collect all responses across all contexts
- Calculate Distinct-1 and Distinct-2

### Step 3: Evaluate Coherence

For each generated response:
1. Construct coherence evaluation prompt
2. Query Llama chat model
3. Extract coherence rating (1-10)
4. If rating < 6, re-evaluate with GPT-4
5. Record final coherence score

### Step 4: Aggregate Metrics

**Summary Statistics**:
- Average semantic diversity across contexts
- Global Distinct-1 and Distinct-2
- Average coherence score
- Incoherence rate (% with score ≤ 5)
- Median, min, max for all metrics

## Implementation Notes for HELM

### Multi-Response Generation

Models must generate **multiple responses per context** (not a single response):
- Default: 5 responses (possibility #1 through #5)
- Can be configured: 10 responses for more comprehensive evaluation

### Instance Creation

Each (context, possibility_number) pair creates one HELM Instance:
- 299 test contexts × 5 possibilities = 1,495 total instances (for 5 responses)
- This allows per-response evaluation while maintaining possibility number control

### Diversity Calculation

**Post-processing required**:
- Group responses by original context
- Calculate semantic similarity within each group
- Aggregate diversity scores

**Not a per-instance metric**: Diversity requires comparing multiple responses for the same context

### Reference Handling

Test data includes 10 candidate diverse responses per context:
- These serve as examples of diverse valid responses
- Not direct ground truth (no 1:1 mapping to possibility numbers)
- Can be used for:
  - Semantic similarity baselines
  - Coherence verification
  - Diversity target examples

## Baseline Results (from paper)

**Base Models** (Mistral-7B, Llama-2-7B):
- Semantic diversity: Low (high avg similarity >0.7)
- Distinct-1: ~0.3-0.4
- Distinct-2: ~0.5-0.6
- Coherence: High (~8-9)
- **Issue**: Good coherence but poor diversity

**With PEFT (Possibility Exploration Fine-Tuning)**:
- Semantic diversity: High (low avg similarity <0.5)
- Distinct-1: ~0.5-0.6
- Distinct-2: ~0.7-0.8
- Coherence: Maintained (~8-9)
- Incoherence rate: <10%
- **Improvement**: Best trade-off between diversity and coherence

**Temperature Sampling** (high temperature):
- Semantic diversity: Medium
- Coherence: Variable, often lower
- **Issue**: Inconsistent quality

**Diverse Beam Search**:
- Semantic diversity: Medium
- Coherence: Good
- Latency: High (3-5x slower)
- **Issue**: Computational cost

## Example Evaluation

**Context**: "I have been working in retail while I finish up school, same as you, I suppose." / "What are you studying?"

**Generated Responses**:
1. "I'm studying psychology and human resources."
2. "I'm majoring in computer science and technology."
3. "Business management is my focus area."
4. "I'm pursuing biology and environmental science."
5. "Education and child development are my areas."

**Semantic Diversity**:
- Pairwise similarities: 0.3, 0.25, 0.4, 0.35, 0.3, ... (avg: 0.32)
- Semantic diversity score: 1 - 0.32 = 0.68 (good)

**N-gram Diversity** (across all 299 contexts):
- Distinct-1: 0.55
- Distinct-2: 0.78

**Coherence Scores**: 9, 9, 8, 9, 8 (avg: 8.6, all above threshold)

**Result**: High diversity with maintained coherence ✓

## Notes

- **Task-agnostic framework**: Can be applied to other text generation tasks (story generation, question answering, etc.)
- **No latency increase**: Unlike diverse beam search, generation time is similar to standard sampling
- **Demographic bias reduction**: Diverse responses can help reduce biases in dialogue systems
- **Semantic vs Lexical**: Emphasis on semantic diversity (meaning) over pure lexical diversity (word choice)
- **Possibility number**: Acts as a "seed" for diversity, not a strict ordering (possibility #3 isn't necessarily "3rd most likely")
