"""
HELM Scenario: PACE (Parallel Association Chain Evaluation)

Paper: PACE: Parallel Association Chain Evaluation for Creativity Measurement (2025)
Code: https://github.com/ziliang6/PACE

Description:
PACE measures creative thinking through word association chains. Given a seed word, models
generate parallel association chains where each word is associated with the immediately preceding
word, testing divergent thinking and semantic creativity.

Task Structure:
1. Given a seed word (e.g., "rock"), generate 3 initial words that associate with it
2. For each initial word, generate a chain of 19 additional words (20 total per chain)
3. Each word in the chain should associate ONLY with the immediately previous word

Prompt format (2-step process):
  Step 1: Starting with the word "{seed}", generate three different words that directly
  associate with this initial word only. For each word, provide a brief explanation.

  Step 2: Starting with "{seed}" → "{second_word}", generate a chain of 20 words where
  each new word associates ONLY with the word immediately before it.

Evaluation:
- Type-Token Ratio (TTR): Vocabulary diversity = unique words / total words across 3 chains
- Association Distance: Semantic distance using GloVe word embeddings (higher = more creative)
- Requires GloVe 6B 300d embeddings for distance calculation (external dependency)

Dataset: 110 seed words across 22 semantic categories (5 words per category) from COCA frequency rankings

Fields used: seed, chapter, rank
Fields skipped: None

Note: This benchmark requires structured JSON output parsing and GloVe embeddings for evaluation.
The scenario provides prompts; evaluation requires custom metric implementation.
"""

from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    TEST_SPLIT,
)


class PACEScenario(Scenario):
    """
    PACE (Parallel Association Chain Evaluation) benchmark for measuring creative thinking
    through word association chains.

    Evaluates divergent thinking by having models generate parallel chains of word associations,
    measuring vocabulary diversity (TTR) and semantic creativity (association distance).
    """

    name = "pace"
    description = "ziliang6/PACE"  # Data source (GitHub repo)
    tags = ["creativity", "divergent_thinking", "word_associations", "open_ended"]

    # All 110 seed words from PACE dataset (from cue_words_ids_with_coca_n.xlsx)
    SEED_WORDS = [
        # The physical world
        {"seed": "rock", "chapter": "The physical world", "rank": 963},
        {"seed": "wood", "chapter": "The physical world", "rank": 1572},
        {"seed": "dust", "chapter": "The physical world", "rank": 2635},
        {"seed": "rainbow", "chapter": "The physical world", "rank": 5822},
        {"seed": "headland", "chapter": "The physical world", "rank": 28230},
        # Kinship
        {"seed": "son", "chapter": "Kinship", "rank": 446},
        {"seed": "female", "chapter": "Kinship", "rank": 1571},
        {"seed": "widow", "chapter": "Kinship", "rank": 5255},
        {"seed": "son-in-law", "chapter": "Kinship", "rank": 12921},
        {"seed": "stepdaughter", "chapter": "Kinship", "rank": 22174},
        # Animals
        {"seed": "eagle", "chapter": "Animals", "rank": 3445},
        {"seed": "worm", "chapter": "Animals", "rank": 3638},
        {"seed": "dove", "chapter": "Animals", "rank": 5699},
        {"seed": "falcon", "chapter": "Animals", "rank": 9047},
        {"seed": "caterpillar", "chapter": "Animals", "rank": 13898},
        # The body
        {"seed": "sick", "chapter": "The body", "rank": 1412},
        {"seed": "toe", "chapter": "The body", "rank": 3507},
        {"seed": "blink", "chapter": "The body", "rank": 5227},
        {"seed": "fingernail", "chapter": "The body", "rank": 13033},
        {"seed": "armpit", "chapter": "The body", "rank": 16837},
        # Food and drink
        {"seed": "meal", "chapter": "Food and drink", "rank": 1871},
        {"seed": "pepper", "chapter": "Food and drink", "rank": 3665},
        {"seed": "crush", "chapter": "Food and drink", "rank": 4066},
        {"seed": "pudding", "chapter": "Food and drink", "rank": 7966},
        {"seed": "morsel", "chapter": "Food and drink", "rank": 14989},
        # Clothing and grooming
        {"seed": "spin", "chapter": "Clothing and grooming", "rank": 2897},
        {"seed": "soap", "chapter": "Clothing and grooming", "rank": 3699},
        {"seed": "bracelet", "chapter": "Clothing and grooming", "rank": 8078},
        {"seed": "lipstick", "chapter": "Clothing and grooming", "rank": 10055},
        {"seed": "tuxedo", "chapter": "Clothing and grooming", "rank": 16841},
        # The house
        {"seed": "bed", "chapter": "The house", "rank": 767},
        {"seed": "pole", "chapter": "The house", "rank": 2177},
        {"seed": "ladder", "chapter": "The house", "rank": 4461},
        {"seed": "shutter", "chapter": "The house", "rank": 7782},
        {"seed": "doorbell", "chapter": "The house", "rank": 17155},
        # Agriculture and vegetation
        {"seed": "grass", "chapter": "Agriculture and vegetation", "rank": 2490},
        {"seed": "mushroom", "chapter": "Agriculture and vegetation", "rank": 5793},
        {"seed": "bamboo", "chapter": "Agriculture and vegetation", "rank": 6902},
        {"seed": "thorn", "chapter": "Agriculture and vegetation", "rank": 7230},
        {"seed": "blossom", "chapter": "Agriculture and vegetation", "rank": 7623},
        # Basic actions and technology
        {"seed": "strike", "chapter": "Basic actions and technology", "rank": 1285},
        {"seed": "broken", "chapter": "Basic actions and technology", "rank": 1774},
        {"seed": "cord", "chapter": "Basic actions and technology", "rank": 4237},
        {"seed": "hack", "chapter": "Basic actions and technology", "rank": 5743},
        {"seed": "scoop", "chapter": "Basic actions and technology", "rank": 8859},
        # Motion
        {"seed": "push", "chapter": "Motion", "rank": 724},
        {"seed": "lift", "chapter": "Motion", "rank": 1664},
        {"seed": "swim", "chapter": "Motion", "rank": 2722},
        {"seed": "glide", "chapter": "Motion", "rank": 6263},
        {"seed": "tumble", "chapter": "Motion", "rank": 7652},
        # Possession
        {"seed": "seek", "chapter": "Possession", "rank": 756},
        {"seed": "hire", "chapter": "Possession", "rank": 2289},
        {"seed": "possess", "chapter": "Possession", "rank": 4336},
        {"seed": "beg", "chapter": "Possession", "rank": 4445},
        {"seed": "squander", "chapter": "Possession", "rank": 16332},
        # Spatial relations
        {"seed": "center", "chapter": "Spatial relations", "rank": 367},
        {"seed": "ball", "chapter": "Spatial relations", "rank": 970},
        {"seed": "collect", "chapter": "Spatial relations", "rank": 1558},
        {"seed": "heap", "chapter": "Spatial relations", "rank": 5664},
        {"seed": "flank", "chapter": "Spatial relations", "rank": 7566},
        # Quantity
        {"seed": "piece", "chapter": "Quantity", "rank": 584},
        {"seed": "count", "chapter": "Quantity", "rank": 1293},
        {"seed": "pair", "chapter": "Quantity", "rank": 1915},
        {"seed": "scarce", "chapter": "Quantity", "rank": 6566},
        {"seed": "trio", "chapter": "Quantity", "rank": 11165},
        # Time
        {"seed": "month", "chapter": "Time", "rank": 249},
        {"seed": "summer", "chapter": "Time", "rank": 1185},
        {"seed": "yesterday", "chapter": "Time", "rank": 1608},
        {"seed": "dusk", "chapter": "Time", "rank": 6831},
        {"seed": "fortnight", "chapter": "Time", "rank": 15308},
        # Sense perception
        {"seed": "dark", "chapter": "Sense perception", "rank": 933},
        {"seed": "dry", "chapter": "Sense perception", "rank": 1655},
        {"seed": "rough", "chapter": "Sense perception", "rank": 2608},
        {"seed": "transparent", "chapter": "Sense perception", "rank": 4876},
        {"seed": "translucent", "chapter": "Sense perception", "rank": 17407},
        # Emotions and values
        {"seed": "pain", "chapter": "Emotions and values", "rank": 943},
        {"seed": "correct", "chapter": "Emotions and values", "rank": 1433},
        {"seed": "anxiety", "chapter": "Emotions and values", "rank": 3294},
        {"seed": "timid", "chapter": "Emotions and values", "rank": 8904},
        {"seed": "fret", "chapter": "Emotions and values", "rank": 9682},
        # Cognition
        {"seed": "seem", "chapter": "Cognition", "rank": 181},
        {"seed": "explain", "chapter": "Cognition", "rank": 856},
        {"seed": "reflect", "chapter": "Cognition", "rank": 1969},
        {"seed": "ponder", "chapter": "Cognition", "rank": 7956},
        {"seed": "muse", "chapter": "Cognition", "rank": 9950},
        # Speech and language
        {"seed": "speak", "chapter": "Speech and language", "rank": 337},
        {"seed": "refuse", "chapter": "Speech and language", "rank": 1575},
        {"seed": "confess", "chapter": "Speech and language", "rank": 4464},
        {"seed": "utter", "chapter": "Speech and language", "rank": 5140},
        {"seed": "eloquent", "chapter": "Speech and language", "rank": 12012},
        # Social and political relations
        {"seed": "subject", "chapter": "Social and political relations", "rank": 803},
        {"seed": "neighbor", "chapter": "Social and political relations", "rank": 2251},
        {"seed": "plot", "chapter": "Social and political relations", "rank": 2694},
        {"seed": "betray", "chapter": "Social and political relations", "rank": 6086},
        {"seed": "shun", "chapter": "Social and political relations", "rank": 11031},
        # Warfare and hunting
        {"seed": "peace", "chapter": "Warfare and hunting", "rank": 1078},
        {"seed": "defeat", "chapter": "Warfare and hunting", "rank": 2753},
        {"seed": "bow", "chapter": "Warfare and hunting", "rank": 3230},
        {"seed": "siege", "chapter": "Warfare and hunting", "rank": 7613},
        {"seed": "spear", "chapter": "Warfare and hunting", "rank": 8065},
        # Law
        {"seed": "murder", "chapter": "Law", "rank": 2756},
        {"seed": "judgment", "chapter": "Law", "rank": 3110},
        {"seed": "punishment", "chapter": "Law", "rank": 5130},
        {"seed": "bribe", "chapter": "Law", "rank": 9116},
        {"seed": "acquit", "chapter": "Law", "rank": 20344},
        # Religion and belief
        {"seed": "pray", "chapter": "Religion and belief", "rank": 2070},
        {"seed": "temple", "chapter": "Religion and belief", "rank": 3670},
        {"seed": "fairy", "chapter": "Religion and belief", "rank": 5222},
        {"seed": "phantom", "chapter": "Religion and belief", "rank": 12241},
        {"seed": "portent", "chapter": "Religion and belief", "rank": 26326},
    ]

    def __init__(self, category: str = "all"):
        """
        Args:
            category: Which semantic category of seed words to use
                - "all": All 110 seed words (default)
                - "common": Words with COCA rank < 5000 (more frequent, 71 words)
                - "rare": Words with COCA rank >= 5000 (less frequent, 39 words)
                - Or specific chapter name (e.g., "Animals", "Cognition")
        """
        super().__init__()
        self.category = category

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generate instances for PACE benchmark."""

        # Filter seed words based on category
        if self.category == "all":
            seeds = self.SEED_WORDS
        elif self.category == "common":
            seeds = [s for s in self.SEED_WORDS if s["rank"] < 5000]
        elif self.category == "rare":
            seeds = [s for s in self.SEED_WORDS if s["rank"] >= 5000]
        else:
            # Filter by chapter name
            seeds = [s for s in self.SEED_WORDS if s["chapter"] == self.category]
            if not seeds:
                raise ValueError(f"Unknown category: {self.category}")

        instances = []
        for seed_info in seeds:
            seed = seed_info["seed"]

            # Create prompt following PACE methodology
            # Step 1: Generate 3 initial associations
            # Step 2: For each, generate a 19-word chain (done in separate calls)
            prompt = f"""Starting with the word "{seed}", generate three different words that directly associate with this initial word only (not with each other). Please put down only single words, and do not use proper nouns (such as names, brands, etc.). For each word, provide a brief explanation of its connection to "{seed}".

Then, for EACH of those three words, generate a chain of 19 additional words where each new word should be associated with ONLY the word immediately before it. Each chain should have 20 total words (including the initial association word).

Format your response as:
Chain 1: [word1] (reason) → [word2] (reason) → ... → [word20] (reason)
Chain 2: [word1] (reason) → [word2] (reason) → ... → [word20] (reason)
Chain 3: [word1] (reason) → [word2] (reason) → ... → [word20] (reason)"""

            # For PACE, references are empty since evaluation requires
            # GloVe embeddings and custom TTR/distance metrics
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[Reference(output=Reference.CORRECT_TAG, tags=[])],
                    split=TEST_SPLIT,
                )
            )

        return instances
