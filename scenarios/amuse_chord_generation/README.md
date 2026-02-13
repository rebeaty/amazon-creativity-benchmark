# Amuse: Chord Progression Generation Diversity Evaluation

**Paper:** Amuse: Human-AI Collaborative Songwriting with Multimodal Inspirations
**Venue:** CHI 2025
**arXiv:** [2412.18940](https://arxiv.org/abs/2412.18940)
**Code:** [GitHub](https://github.com/elianakim/Amuse)
**Project Page:** [https://yewon-kim.com/amuse/](https://yewon-kim.com/amuse/)

## Overview

Amuse evaluates LLMs on generating **diverse and musically coherent chord progressions** from music keywords. This benchmark is unique in that it measures **diversity as a primary quality** - not accuracy against ground truth, but the variety and richness of generated outputs.

## Task Description

**Input:** Music keyword(s) + Musical parameters (key, mode, bars)

**Output:** Multiple diverse chord progressions (typically 30)

**Example:**
```
Input:
  Keywords: "dreamy, jazz, soft"
  Key: B
  Mode: Maj
  Bars: 4

Output (30 progressions):
  C#m7 F#7 Bmaj9 d#dim/C
  Emaj7 A#m7b5 D#m7 G#7
  Bmaj7 G#m7 C#7 F#maj7
  ...
  (27 more progressions)
```

## Dataset

**Music Keywords:** 254 keywords from suno.wiki

**Categories:**
- **Genres:** Jazz, Rock, EDM, Classical, Hip-Hop, etc.
- **Moods:** Happy, Sad, Dramatic, Ethereal, Groovy, etc.
- **Styles:** Ambient, Syncopated, Progressive, Intimate, etc.
- **Instrumentation:** Acoustic Guitar, Bass, Percussion, etc.
- **Eras/Styles:** 1960s, Broadway, Barbershop, etc.

**Musical Parameters:**
- **Keys:** C, G, D, A, E, B, F#, Db, Ab, Eb, Bb, F
- **Modes:** maj, min, dor, phr, lyd, mix, loc, hmin, phdm
- **Bars:** Typically 4 chords per progression

**Total Instances:** 254 (one per keyword, with randomly sampled key/mode)

## Evaluation Metrics

### 1. Self-BLEU (Diversity) ⭐ Primary Metric

**Purpose:** Measure diversity among generated progressions
**Interpretation:** **Lower is better** (opposite of standard BLEU!)

**Method:**
- Generate 30 progressions per keyword
- For each progression, compute BLEU using other 29 as references
- Average across all progressions
- Repeat 100 times, report mean ± std

**Paper Results:**
- **Amuse (batch prompting):** 0.30 ± 0.12 ✓ More diverse
- **Baseline (iterative):** 0.61 ± 0.18 ✗ Less diverse

### 2. Jensen-Shannon Divergence (Coherence)

**Purpose:** Measure similarity to real music
**Interpretation:** Lower = closer to real music distribution

**Method:**
- Compute chord n-gram distributions (unigram & bigram)
- Compare to Hooktheory dataset (26,175 real songs)
- Calculate JSD between distributions

**Paper Results:**
- **Amuse unigram JSD:** 0.27
- **Amuse bigram JSD:** 0.46

### 3. Human Evaluation (Optional)

- 45 musicians evaluated musical coherence and keyword relevance
- 900 pairwise comparisons
- Amuse favored for keyword relevance (58%), tied on coherence

## Key Findings from Paper

1. **Batch Prompting > Iterative Prompting**
   - Generating 30 progressions in one prompt produces **2x more diversity** (Self-BLEU: 0.30 vs 0.61)
   - Challenges the common practice of querying LLMs iteratively

2. **Rejection Sampling Improves Coherence**
   - Filtering LLM outputs with a music prior model reduces JSD
   - Maintains diversity while improving musical plausibility

3. **Multimodal Inspiration Works**
   - System successfully converts images/audio → keywords → chords
   - Enables creative songwriting from non-textual inspiration

## Chord Format

Chords use standard music notation with components:

1. **Root Note:** A-G with accidentals (#, b, x)
2. **Quality:** maj, min, aug, dim
3. **Extensions:** 6/9, 7, 9, 11, 13
4. **Suspensions:** sus2, sus4, sus#2, sus#4
5. **Added Notes:** add2, add4, add6, add9, add11, add13
6. **Alterations:** b5, #5, b9, #9, #11, b13
7. **Slash Chords:** /E, /G#, /Bb (alternate bass)

**Valid Examples:**
- `C` (simple major)
- `Am7` (A minor seventh)
- `Cmaj9` (C major ninth)
- `Dsus4/F#` (D suspended 4th over F# bass)
- `G#m7b5` (G# half-diminished)

**Output Format:**
```
C Em F G
Am7 Dm7 G7 Cmaj7
Fmaj7 Bm7b5 E7 Am
...
(one progression per line, chords separated by spaces)
```

## Implementation Notes

### Scenario Features

- **254 test instances** (one per music keyword)
- **Batch prompting mode** (default): Generate 30 progressions at once
- **Single prompting mode** (optional): Generate 1 progression per query
- **Random sampling** of musical parameters (key, mode) per keyword
- **No ground truth references** - evaluation via distributional metrics

### Requirements for Evaluation

**Custom Metrics Required:**
- `ChordDiversityMetric` - Implements Self-BLEU calculation
- `ChordCoherenceMetric` - Implements JSD with Hooktheory data

**External Data:**
- **Hooktheory dataset:** 26,175 songs with chord progressions
  - Download: `https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz`
  - Filter: Examples with 'HARMONY' tag
  - Processing: Transpose to C, extract 4+ chord progressions

See `metric_notes.md` for detailed implementation instructions.

### Python Syntax

```bash
python -m py_compile scenarios/amuse_chord_generation/scenario.py
# ✓ Valid syntax
```

## Comparison to Other Benchmarks

| Aspect | Amuse | Traditional Benchmarks |
|--------|-------|----------------------|
| **Ground Truth** | None | Fixed correct answers |
| **Metric** | Diversity (Self-BLEU) | Accuracy (Exact Match, F1) |
| **Evaluation** | Distributional | Instance-level |
| **Goal** | Maximize variety | Minimize error |
| **Interpretation** | Lower BLEU = better | Higher accuracy = better |

## Related Work

- **Hooktheory Dataset:** SheetSage (arXiv:2212.01884) - Source of real chord progressions
- **Self-BLEU:** Texygen (Zhu et al., 2018) - Diversity metric for text generation
- **Chord Generation:** CrystaLLM, MusicGen-Chord - Other symbolic music generation systems

## Citation

```bibtex
@article{kim2024amuse,
    title={Amuse: Human-AI Collaborative Songwriting with Multimodal Inspirations},
    author={Kim, Yewon and Lee, Sung-Ju and Donahue, Chris},
    year={2024},
    journal={arXiv preprint arXiv:2412.18940},
}
```

## Future Work

- **Multiple Keywords:** Extend to multi-keyword combinations (e.g., "sad jazz piano")
- **Longer Progressions:** Evaluate 8-bar or full song structures
- **Audio Synthesis:** Integrate with audio rendering evaluation
- **Human Correlation:** Study correlation between Self-BLEU and human preference

## Links

- 📄 [Paper (arXiv)](https://arxiv.org/abs/2412.18940)
- 💻 [Code (GitHub)](https://github.com/elianakim/Amuse)
- 🌐 [Project Page](https://yewon-kim.com/amuse/)
- 🎵 [Audio Examples](https://yewon-kim.com/amuse/#listening-study)
