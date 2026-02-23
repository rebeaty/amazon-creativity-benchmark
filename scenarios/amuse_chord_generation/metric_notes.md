# Evaluation Metrics: Amuse Chord Progression Generation

**Source:** Amuse: Human-AI Collaborative Songwriting with Multimodal Inspirations (CHI 2025)
**Paper:** [arXiv:2412.18940](https://arxiv.org/abs/2412.18940)
**Code:** [GitHub](https://github.com/elianakim/Amuse)

## Overview

The Amuse benchmark evaluates chord progression generation using **custom diversity and coherence metrics**. Standard text metrics (BLEU, ROUGE) are inappropriate because they would penalize diversity, which is the primary quality being measured.

## Required Evaluation Metrics

### 1. Self-BLEU (Diversity)

**Purpose:** Measure diversity of generated chord progressions
**Interpretation:** **Lower scores = more diverse** (opposite of standard BLEU)

**Methodology:**
- For each keyword, generate 30 four-bar chord progressions
- For each progression i, compute BLEU score using the other 29 as references
- Average BLEU scores across all 30 progressions
- Repeat for 100 different keyword sets and report mean ± std

**Paper Results:**
- Amuse (batch prompting): **0.30 ± 0.12** (more diverse)
- Baseline (conventional): **0.61 ± 0.18** (less diverse)

**Implementation:**
```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_self_bleu(progressions: List[str]) -> float:
    """
    Compute Self-BLEU for a set of chord progressions.

    Args:
        progressions: List of chord progression strings (e.g., "C Em F G")

    Returns:
        Self-BLEU score (0-1, lower = more diverse)
    """
    # Tokenize each progression
    tokenized = [prog.split() for prog in progressions]

    scores = []
    smoothing = SmoothingFunction().method1

    for i, hypothesis in enumerate(tokenized):
        # Use all other progressions as references
        references = [tokenized[j] for j in range(len(tokenized)) if j != i]

        # Compute BLEU score
        score = sentence_bleu(
            references,
            hypothesis,
            smoothing_function=smoothing
        )
        scores.append(score)

    return np.mean(scores)
```

### 2. Jensen-Shannon Divergence (Coherence)

**Purpose:** Measure how similar generated progressions are to real music
**Interpretation:** Lower scores = closer to real music distribution

**Methodology:**
- Compute chord n-gram distributions for generated progressions
- Compute chord n-gram distributions for Hooktheory dataset (ground truth)
- Calculate JSD between the two distributions
- Report both unigram and bigram JSD

**Paper Results:**
- Amuse unigram JSD: **0.27**
- Amuse bigram JSD: **0.46**

**Implementation:**
```python
from scipy.spatial.distance import jensenshannon
import pandas as pd

def compute_jsd(generated_progs: List[List[str]],
                reference_progs: List[List[str]],
                mode: str = 'unigram') -> float:
    """
    Compute Jensen-Shannon Divergence between generated and reference progressions.

    Args:
        generated_progs: List of tokenized chord progressions
        reference_progs: List of tokenized chord progressions from Hooktheory
        mode: 'unigram' or 'bigram'

    Returns:
        JSD value (0-1, lower = more similar to real music)
    """
    def get_distribution(progressions, mode):
        if mode == 'unigram':
            # Flatten all chords
            chords = [chord for prog in progressions for chord in prog]
        elif mode == 'bigram':
            # Extract bigrams
            bigrams = []
            for prog in progressions:
                bigrams.extend(zip(prog[:-1], prog[1:]))
            chords = bigrams
        else:
            raise ValueError("Mode must be 'unigram' or 'bigram'")

        # Get normalized distribution
        freq = pd.Series(chords).value_counts(normalize=True)
        return freq

    gen_dist = get_distribution(generated_progs, mode)
    ref_dist = get_distribution(reference_progs, mode)

    # Align distributions (fill missing chords with 0)
    all_chords = set(gen_dist.index) | set(ref_dist.index)
    gen_aligned = [gen_dist.get(c, 0) for c in all_chords]
    ref_aligned = [ref_dist.get(c, 0) for c in all_chords]

    # Compute JSD
    jsd = jensenshannon(ref_aligned, gen_aligned)
    return jsd
```

### 3. Human Evaluation (Optional)

The paper includes human listening studies with 45 musicians evaluating:
- **Musical coherence:** Does the progression sound musically logical?
- **Keyword relevance:** Does the progression match the keyword?

**Methodology:**
- Pairwise comparisons between model outputs
- 900 total judgments (150 per keyword set × 3 listeners × 2 conditions)
- Metrics: Win rate, preference percentage

This is optional for automated HELM evaluation but valuable for full validation.

## Required Data

### Hooktheory Dataset (for JSD calculation)

The reference dataset for computing coherence metrics:

**Download:**
```bash
wget https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz
```

**Processing:**
1. Filter examples with 'HARMONY' tag
2. Extract chord progressions
3. Transpose all to C major/minor for consistency
4. Filter progressions with at least 4 chords

**Statistics:**
- Total songs: 26,175
- Filtered progressions: ~50 hours of chord data
- Genres: Pop, rock, EDM, jazz, classical

### Music Keywords

254 keywords from suno.wiki covering:
- Genres: Jazz, Rock, EDM, Classical, etc.
- Moods: Happy, Sad, Dramatic, Ethereal, etc.
- Styles: Groovy, Syncopated, Ambient, etc.
- Instrumentation: Acoustic Guitar, Bass, etc.

Available at: `https://raw.githubusercontent.com/elianakim/Amuse/main/assets/music_keywords.txt`

## Metric Configuration for HELM

To implement these metrics in HELM:

### 1. Create Custom Metric Classes

```python
# helm/benchmark/metrics/chord_diversity_metric.py
class ChordDiversityMetric(Metric):
    """Self-BLEU metric for chord progression diversity."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        # Generate 30 progressions per instance
        # Compute Self-BLEU
        # Return diversity score

# helm/benchmark/metrics/chord_coherence_metric.py
class ChordCoherenceMetric(Metric):
    """JSD metric for chord progression coherence."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        # Load Hooktheory reference data
        # Compute unigram and bigram JSD
        # Return coherence scores
```

### 2. Create RunSpec

```python
def get_amuse_chord_generation_metric_specs():
    return [
        MetricSpec(
            class_name="helm.benchmark.metrics.chord_diversity_metric.ChordDiversityMetric",
            args={
                "num_progressions": 30,
                "num_trials": 100,
            },
        ),
        MetricSpec(
            class_name="helm.benchmark.metrics.chord_coherence_metric.ChordCoherenceMetric",
            args={
                "hooktheory_path": "./data/Hooktheory.json.gz",
            },
        ),
    ]
```

## Output Format

**Expected LLM output for batch prompting:**
```
C Em F G
Am F C G
Dm G Em Am
...
(30 progressions total, one per line)
```

**Parsing Requirements:**
- Split output by newlines
- Each line is one progression
- Split by spaces to get individual chords
- Validate chord format (root note + optional quality/extensions)

## Notes

- **Diversity vs. Quality Trade-off:** Higher diversity (lower Self-BLEU) doesn't necessarily mean better quality. Both diversity and coherence (JSD) must be evaluated.
- **Hooktheory Required:** JSD calculation requires the Hooktheory dataset as reference distribution.
- **Prompt Engineering Matters:** The paper shows batch prompting significantly increases diversity compared to iterative querying.
- **Stochasticity:** Results will vary across runs due to sampling. Paper reports mean ± std over 100 trials.

## References

- Amuse paper: [arXiv:2412.18940](https://arxiv.org/abs/2412.18940)
- Hooktheory dataset paper: [SheetSage (arXiv:2212.01884)](https://arxiv.org/abs/2212.01884)
- Self-BLEU reference: Zhu et al., "Texygen: A Benchmarking Platform for Text Generation Models" (2018)
