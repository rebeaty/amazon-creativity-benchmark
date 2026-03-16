# Annotator Requirements: EQBench Creative Writing v3

Source: https://github.com/EQ-bench/creative-writing-bench
Leaderboard: https://eqbench.com/creative_writing_longform.html
Author: Samuel J. Paech

## Configuration for LLMAsJuryAnnotator

**Judge model**: Claude Sonnet 4 (or compatible LLM)
**Task**: Pairwise comparison of creative writing outputs
**Dimensions**: Multiple criteria (see below)
**Scale**: Winner selection + quality scores

## Generation Settings

Before evaluation, models generate responses with:
- **Temperature**: 0.7
- **Min-p**: 0.1
- **Iterations**: 3 per prompt (96 total outputs for 32 prompts)
- **Target length**: ~1000 words per output

## Evaluation Criteria

Outputs are judged on multiple dimensions:

1. **Character Authenticity**
   - Believable motivations and personalities
   - Consistent characterization
   - Depth and complexity

2. **Originality**
   - Fresh perspectives and unique angles
   - Avoidance of clichés and "AI slop"
   - Creative plot elements

3. **Plot Coherence**
   - Logical progression
   - Clear cause and effect
   - Satisfying scene structure

4. **Emotional Engagement**
   - Evocative descriptions
   - Reader investment in characters
   - Effective use of tension

5. **Prose Quality**
   - Vivid sensory details
   - Varied sentence structure
   - Strong voice and style

## Judge Prompt Template (Pairwise Comparison)

```
You will be comparing two creative writing samples based on the following prompt:

{writing_prompt}

Sample A:
{output_a}

Sample B:
{output_b}

Please evaluate both samples on these criteria:
1. Character authenticity and depth
2. Originality and creativity
3. Plot coherence and structure
4. Emotional engagement
5. Prose quality and style

Which sample is better overall? Explain your reasoning and select the winner.

Output format:
Winner: [A/B]
Reasoning: [explanation]
```

## Metrics

### 1. Elo Rating (Primary Metric)
- **Calculation**: Glicko rating system from pairwise comparisons
- **Process**: All model outputs compared head-to-head across same prompts
- **Interpretation**: Higher Elo = consistently beats other models in comparisons
- **Starting**: 1500 Elo baseline

### 2. Rubric Score (Secondary Metric)
- **Range**: 1-10 scale per criterion
- **Calculation**: Average across all 5 criteria
- **Aggregation**: Mean score across all 96 outputs (3 per prompt × 32 prompts)

## Prompt Categories

The 32 prompts span diverse genres:
- Historical fiction
- Science fiction
- Fantasy
- Romance
- Humor
- Literary fiction
- Horror
- Mystery
- Contemporary fiction

Each prompt includes:
- **Base instruction**: Genre and scenario
- **Seed modifiers**: 10 optional variations for diversity

## Slop Detection

The benchmark includes automated "slop" detection to penalize generic AI writing:
- **Slop phrases**: Common AI-generated clichés tracked in `slop_list.json`
- **Bigrams/Trigrams**: Multi-word AI patterns (`slop_list_bigrams.json`, `slop_list_trigrams.json`)
- **Penalty**: Frequency of slop phrases reduces quality score

## Notes for HELM Adaptation

- **Iteration handling**: Create 3 instances per prompt (96 total)
- **Generation params**: Pass temperature=0.7, min_p=0.1 to models
- **Pairwise setup**: Each model output compared against all others for same prompt
- **Judge model**: Claude Sonnet 4 recommended (or GPT-4-turbo equivalent)
- **Empty references**: No ground-truth references (pure generation task)
- **Elo calculation**: Requires custom metric implementing Glicko system

## Human Correlation

- Benchmark designed to emphasize challenging prompts that expose weaknesses
- Focus on subtle creative elements (humor, unusual perspectives, romance)
- Intended to be "hard" - models typically score lower than on general benchmarks

## References

- GitHub: https://github.com/EQ-bench/creative-writing-bench
- Leaderboard: https://eqbench.com/creative_writing_longform.html
- DARLING usage: arXiv:2509.02534 (creative writing evaluation)
- Criteria: `data/creative_writing_criteria.txt`
- Judge prompt: `data/pairwise_prompt.txt`
