"""
HELM Scenario: S-DAT (Synthetic-Divergent Association Task)

Paper: S-DAT: A Multilingual, GenAI-Driven Framework for Automated Divergent Thinking Assessment
       https://arxiv.org/abs/2505.09068
       Haase, J., Hanel, P. H. P., & Pokutta, S. (2025). AAAI/ACM AIES.

Original DAT: https://www.pnas.org/doi/10.1073/pnas.2022340118
              Olson, J. A., et al. (2021). PNAS.

Code: https://sdat.iol.zib.de/ (online assessment tool)
      https://osf.io/pv84c/ (data and analysis code)

Task: Generate 10 semantically diverse words to measure divergent thinking.
      Participants generate nouns that are as unrelated to each other as possible.
      S-DAT extends the original DAT to support multilingual assessment across 11+ languages.

Prompt format: From original DAT (Olson et al., 2021), used in S-DAT framework
  "Please enter 10 words that are as different from each other as possible, in all
  meanings and uses of the words. Rules: Only single words in English. Only nouns
  (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or
  places). No specialised vocabulary (e.g., no technical terms). Think of the words
  on your own (e.g., do not just look at objects in your surroundings). Make a list
  of these 10 words, a single word in each entry of the list."

Evaluation: Requires custom semantic distance metric (see metric_notes.md)
  - Average cosine dissimilarity between all word pairs (45 pairs from 10 words)
  - S-DAT uses IBM granite-embedding-278m-multilingual embeddings (278M parameters)
  - Original DAT uses GloVe embeddings (monolingual English)
  - Score calibrated to human distribution from Olson et al. data (N=8,572)
  - Percentiles: 5%=72.17, 25%=76.44, 50%=79.11, 75%=82.03, 95%=86.59

Fields used: Fixed prompt (no input dataset)
Fields skipped: N/A (generation task)

Notes:
  - S-DAT supports 11 languages: English, Spanish, German, Russian, Hindi, Japanese,
    French, Italian, Dutch, Portuguese, Polish, plus Arabic, Czech, Korean, Chinese
  - This scenario uses English prompts; multilingual variants can be added
  - 100 instances provide statistical reliability while balancing efficiency
  - Correlation with original DAT: r=.60-.67 (Studies 1a, 1b, 2)
  - Shows convergent validity with AUT (r=.13-.27) and discriminant validity with
    convergent thinking tasks (Bridge-the-Associative-Gap: r=.08-.11, non-significant)
"""

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    TEST_SPLIT,
)


class SDATScenario(Scenario):
    """
    S-DAT (Synthetic-Divergent Association Task) - Multilingual divergent thinking assessment.

    The Divergent Association Task measures creativity through semantic diversity of
    generated words. S-DAT extends the original DAT (Olson et al., 2021) to support
    multilingual assessment using IBM's granite-embedding-278m-multilingual embeddings.
    Higher average semantic distance between words indicates greater divergent thinking ability.

    Paper: Haase, Hanel, & Pokutta (2025). AAAI/ACM AIES.
    """

    name = "sdat"
    description = "S-DAT: Multilingual divergent thinking assessment (Haase et al., 2025)"
    tags = ["creativity", "divergent_thinking", "generation", "multilingual"]

    # Prompt from original DAT (Olson et al., 2021), used in S-DAT framework
    # S-DAT applies this task across 11+ languages using multilingual embeddings
    BASE_PROMPT = (
        "Please enter 10 words that are as different from each other as possible, "
        "in all meanings and uses of the words. Rules: Only single words in English. "
        "Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no "
        "specific people or places). No specialised vocabulary (e.g., no technical "
        "terms). Think of the words on your own (e.g., do not just look at objects "
        "in your surroundings). Make a list of these 10 words, a single word in each "
        "entry of the list."
    )

    def get_instances(self, output_path: str) -> list[Instance]:
        """
        Generate instances for S-DAT evaluation.

        Creates 100 instances with identical prompts. Multiple instances enable:
        1. Statistical reliability (measuring consistency of divergent thinking)
        2. Aggregation of semantic distance scores
        3. Comparison with human baseline data (N=8,572 from Olson et al., 2021)

        Returns:
            List of Instance objects with the DAT prompt and no references.
        """
        instances = []

        # Create 100 instances for statistical validity
        # S-DAT was validated using large-scale datasets; 100 provides good balance for HELM
        for i in range(100):
            instances.append(
                Instance(
                    input=Input(text=self.BASE_PROMPT),
                    references=[],  # No ground truth; evaluated by semantic distance metric
                    split=TEST_SPLIT,
                )
            )

        return instances
