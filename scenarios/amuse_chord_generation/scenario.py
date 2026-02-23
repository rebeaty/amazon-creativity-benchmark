"""
HELM Scenario: Amuse - Chord Progression Generation Diversity Evaluation

Paper: Amuse: Human-AI Collaborative Songwriting with Multimodal Inspirations (CHI 2025)
arXiv: https://arxiv.org/abs/2412.18940
Code: https://github.com/elianakim/Amuse

Task: Generate diverse 4-bar chord progressions from music keywords.

The paper evaluates two prompting approaches:
1. Batch prompting: Generate 30 progressions in one prompt (more diverse)
2. Conventional prompting: Query LLM 30 times separately (less diverse)

Evaluation uses Self-BLEU to measure diversity (lower = more diverse):
- Amuse (batch): 0.30±0.12
- Baseline (conventional): 0.61±0.18

Additional metrics:
- Jensen-Shannon Divergence (JSD) against real music data (Hooktheory)
- Human evaluation for musical coherence and keyword relevance

Prompt format:
  You are a musical assistant generating chord progressions based on keywords.

  Keywords: {keywords}
  Key: {key}
  Mode: {mode}
  Bars: {bars}

  Generate {num_progressions} diverse chord progressions...

Fields used: music_keywords (254 keywords from suno.wiki)
Fields skipped: None - generative task without ground truth progressions

Evaluation: Custom metrics required - Self-BLEU for diversity, JSD for coherence.
            See metric_notes.md for implementation details.
"""

from typing import List
import random
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded
import os


class AmuseChordGenerationScenario(Scenario):
    """
    Amuse Chord Progression Generation - Diversity Evaluation

    Evaluates LLMs on generating diverse chord progressions from music keywords.
    The task tests whether batch prompting (generating multiple progressions at once)
    produces more diverse outputs than conventional iterative prompting.

    Dataset: 254 music keywords from suno.wiki covering genres, styles, moods,
    and instrumentation.
    """

    name = "amuse_chord_generation"
    description = "Music chord progression generation from keywords (Amuse CHI 2025)"
    tags = ["creativity", "music_generation", "chord_progression", "diversity"]

    # Standard musical parameters used in the paper
    KEYS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
    MODES = ["Maj", "Min", "Dor", "Phr", "Lyd", "Mix", "Loc", "Hmin", "Phdm"]  # Capitalized as in paper
    DEFAULT_BARS = 4
    DEFAULT_NUM_PROGRESSIONS = 30  # Paper generates 30 for diversity evaluation

    KEYWORDS_URL = "https://raw.githubusercontent.com/elianakim/Amuse/main/assets/music_keywords.txt"

    # Exact prompt from paper Appendix (label: appendix:prompts:chordsamuse)
    SYSTEM_PROMPT = """You are a musical assistant generating chord progressions based on user-provided keywords, key, mode, and bar. The keywords describe the genre, style, and song type. The key specifies a root note that is in [C, G, D, A, E, B, F#, Db, Ab, Eb, Bb, F]. The mode specifies a scale that is in [Maj, Min, Dor, Phr, Lyd, Mix, Loc, Hmin, Phdm]. The bar specifies the number of chords to generate for each progression. Your task is to create {num_progressions} diverse chord progressions conforming to the keywords, key, and mode. Each progression should consist of the same number of chords as the bar input, with each chord separated by a space ' ' and each progression on a new line.

Instructions:
1. Analyze Chord Functions: Determine the functions of chords in the given key and mode. Tonic (I, vi) provides resolution and stability. Subdominant (IV, ii) creates movement away from the tonic. Dominant (V, vii°) creates tension that needs to resolve to the tonic.
2. Analyze the Keywords: Determine the chord components and progression patterns based on the keywords. For example, for jazz-related keywords, consider using seventh chords, altered chords, and common jazz progressions like ii-V-I. For keywords like 'sadness' or 'emotional,' use minor chords, diminished chords, and progressions that create tension.
3. Generate {num_progressions} Chord Progressions: Create {num_progressions} distinct chord progressions that fit the specified key and mode and match the keywords. Each progression should be unique and align with the bar parameter (i.e., if bars = 4, each progression should have 4 chords). Ensure diversity by varying the chord components (root, quality, extensions, alterations, etc.), progression patterns (diatonic/chromatic), and cadences.

Each chord text can have the following components, in order:
1. Root Note: A-G, with optional accidentals (#, b, x).
2. Chord Quality: maj, min, aug, dim.
3. Extensions: Specific chord extensions such as 6/9, 7, 9, 11, 13.
4. Suspended Chords: Suspended chords such as sus2, sus4, sus#2, sus#4.
5. Added Notes: Added notes such as add2, add4, add6, add9, add11, add13.
6. Altered Notes: Alterations such as b5, #5, b9, #9, #11, b13.
7. Slash Chords: Alternate bass notes such as /E, /G#, /Bb, /Dx.

Ensure the chord progressions are musically coherent, stylistically appropriate, and diverse. Include extensions, suspensions, adds, altered notes, slash chords as needed to achieve maximum diversity. Use both diatonic and chromatic chords to enhance the progressions. Respond only with the chord progressions, avoiding any additional commentary or formatting.

Examples:
User keywords: dreamy, jazz, soft | Key: B | Mode: Maj | Bars: 4
Example Progressions:
C#m7 F#7 Bmaj9 d#dim/C
Emaj7 A#m7b5 D#m7 G#7
User keywords: singer-songwriter, acoustic, emotional | Key: F# | Mode: Maj | Bars: 3
Example progressions:
F# B/F# C#/G#
C# D#msus2 D#m/
User keywords: orchestral, adventurous, epic | Key: D | Mode: Min | Bars: 4
Example progressions:
dm gm/Bb gm dm
Bb F C C#dim

Generate {num_progressions} progressions for each user input, following the above guidelines and ensuring diversity and musical coherence. Keep the chord format the same as the examples provided (e.g., G, Amaj7, Cm are valid formats, but Gmaj, Cmin are invalid formats)."""

    def __init__(self, num_progressions: int = 30, diversity_eval: bool = True):
        """
        Args:
            num_progressions: Number of progressions to generate per keyword (default: 30)
            diversity_eval: If True, use batch prompting for diversity evaluation
        """
        super().__init__()
        self.num_progressions = num_progressions
        self.diversity_eval = diversity_eval

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Create instances for chord progression generation.

        Each instance corresponds to one music keyword with randomly sampled
        musical parameters (key, mode).
        """

        # Download keywords
        keywords_path = os.path.join(output_path, "music_keywords.txt")
        ensure_file_downloaded(
            source_url=self.KEYWORDS_URL,
            target_path=keywords_path,
            unpack=False,
        )

        # Load keywords
        with open(keywords_path, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(keywords)} music keywords")

        instances = []
        random.seed(42)  # For reproducible key/mode sampling

        for keyword in keywords:
            # Sample random musical parameters (as done in paper evaluation)
            key = random.choice(self.KEYS)
            mode = random.choice(self.MODES)

            # Build prompt
            if self.diversity_eval:
                # Batch prompting: generate multiple progressions at once
                prompt = self._build_batch_prompt(keyword, key, mode)
            else:
                # Conventional prompting: single progression
                # (HELM can call this multiple times for diversity comparison)
                prompt = self._build_single_prompt(keyword, key, mode)

            # Create instance
            # Note: No ground truth reference - this is a generative task evaluated
            # by diversity metrics (Self-BLEU) and coherence (JSD against Hooktheory)
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],  # No reference - evaluation via custom metrics
                    split=TEST_SPLIT,
                )
            )

        return instances

    def _build_batch_prompt(self, keyword: str, key: str, mode: str) -> str:
        """Build prompt for batch generation (Amuse approach)."""
        system = self.SYSTEM_PROMPT.format(num_progressions=self.num_progressions)
        user = f"""User keywords: {keyword}
Key: {key}
Mode: {mode}
Bars: {self.DEFAULT_BARS}

Generate {self.num_progressions} progressions:"""

        return f"{system}\n\n{user}"

    def _build_single_prompt(self, keyword: str, key: str, mode: str) -> str:
        """Build prompt for single progression generation (conventional approach - baseline from paper appendix)."""
        # Exact baseline prompt from paper Appendix (label: appendix:prompts:chordsbaseline)
        system = """You are a musical assistant generating chord progressions based on user-provided keywords, key, mode, and bar. The keywords describe the genre, style, and song type. The key specifies a root note that is in [C, G, D, A, E, B, F#, Db, Ab, Eb, Bb, F]. The mode specifies a scale that is in [Maj, Min, Dor, Phr, Lyd, Mix, Loc, Hmin, Phdm]. The bar specifies the number of chords to generate for each progression. Your task is to create a chord progression conforming to the keywords, key, and mode. The progression should consist of the same number of chords as the bar input, with each chord separated by a space ' ' and the progression on a new line.

Instructions:
1. Analyze Chord Functions: Determine the functions of chords in the given key and mode. Tonic (I, vi) provides resolution and stability. Subdominant (IV, ii) creates movement away from the tonic. Dominant (V, vii°) creates tension that needs to resolve to the tonic.
2. Analyze the Keywords: Determine the chord components and progression patterns based on the keywords. For example, for jazz-related keywords, consider using seventh chords, altered chords, and common jazz progressions like ii-V-I. For keywords like 'sadness' or 'emotional,' use minor chords, diminished chords, and progressions that create tension.
3. Generate a Chord Progression: Create a chord progression that fits the specified key and mode and matches the keywords. The progression should align with the bar parameter (i.e., if bars = 4, the progression should have 4 chords).

Each chord text can have the following components, in order:
1. Root Note: A-G, with optional accidentals (#, b, x).
2. Chord Quality: maj, min, aug, dim.
3. Extensions: Specific chord extensions such as 6/9, 7, 9, 11, 13.
4. Suspended Chords: Suspended chords such as sus2, sus4, sus#2, sus#4.
5. Added Notes: Added notes such as add2, add4, add6, add9, add11, add13.
6. Altered Notes: Alterations such as b5, #5, b9, #9, #11, b13.
7. Slash Chords: Alternate bass notes such as /E, /G#, /Bb, /Dx.

Ensure the chord progression is musically coherent and stylistically appropriate. Include extensions, suspensions, adds, altered notes, and slash chords as needed to achieve a rich and satisfying progression. Use both diatonic and chromatic chords to enhance the progression. Respond only with the chord progression, avoiding any additional commentary or formatting.

Examples:
User keywords: dreamy, jazz, soft | Key: B | Mode: Maj | Bars: 4
Example Progression:
C#m7 F#7 Bmaj9 d#dim/C
User keywords: singer-songwriter, acoustic, emotional | Key: F# | Mode: Maj | Bars: 3
Example progression:
F# B/F# C#/G#
User keywords: orchestral, adventurous, epic | Key: D | Mode: Min | Bars: 4
Example progression:
dm gm/Bb gm dm

Generate a progression for each user input, following the above guidelines and ensuring musical coherence. Keep the chord format the same as the examples provided (e.g., G, Amaj7, Cm are valid formats, but Gmaj, Cmin are invalid formats)."""

        user = f"""User keywords: {keyword}
Key: {key}
Mode: {mode}
Bars: {self.DEFAULT_BARS}

Generate a progression:"""

        return f"{system}\n\n{user}"
