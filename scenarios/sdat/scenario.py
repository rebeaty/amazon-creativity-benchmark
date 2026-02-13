"""
HELM Scenario: S-DAT (Synthetic-Divergent Association Task)

Paper: Divergent Creativity in Humans and Large Language Models
       https://arxiv.org/abs/2405.13012
       Bellemare, A. et al. (2025). Scientific Reports.

S-DAT Framework: https://arxiv.org/abs/2505.09068
                 Haase, J., Hanel, P. H. P., & Pokutta, S. (2025). AAAI/ACM AIES.

Original DAT: https://www.pnas.org/doi/10.1073/pnas.2022340118
              Olson, J. A., et al. (2021). PNAS.

Code: https://github.com/AntoineBellemare/DAT_GPT

Task: Generate 10 semantically diverse words to measure divergent thinking.
      Participants generate nouns that are as unrelated to each other as possible.

Prompt format: Based on DAT_GPT evaluation methodology (Bellemare et al., 2025)
  "Please enter 10 words that are as different from each other as possible, in all
  meanings and uses of the words. Rules: Only single words in English. Only nouns
  (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or
  places). No specialised vocabulary (e.g., no technical terms). Think of the words
  on your own (e.g., do not just look at objects in your surroundings). Make a list
  of these 10 words, a single word in each entry of the list."

Evaluation: Requires custom semantic distance metric (see metric_notes.md)
  - Average cosine dissimilarity between all word pairs (45 pairs from 10 words)
  - S-DAT uses granite-embedding-278m-multilingual embeddings
  - Original DAT uses GloVe embeddings
  - Score calibrated to human distribution (mean=78.5, sd=15.2)

Fields used: Fixed prompt (no input dataset)
Fields skipped: N/A (generation task)

Notes:
  - DAT_GPT study used 500 iterations per model for statistical reliability
  - This scenario uses 100 instances as a balance between validity and efficiency
  - Multiple strategy variations exist (thesaurus, etymology, opposites) but base
    prompt is sufficient for core evaluation
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
    S-DAT (Synthetic-Divergent Association Task) - Divergent thinking assessment.

    The Divergent Association Task measures creativity through semantic diversity of
    generated words. Higher average semantic distance between words indicates greater
    divergent thinking ability.
    """

    name = "sdat"
    description = "Synthetic-Divergent Association Task - divergent thinking assessment"
    tags = ["creativity", "divergent_thinking", "generation"]

    # Prompt from DAT_GPT study (Bellemare et al., 2025)
    # https://github.com/AntoineBellemare/DAT_GPT/blob/main/scripts/api_call_dat_gpt4.py
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
        3. Comparison with human baseline data (n=8,900+)

        Returns:
            List of Instance objects with the DAT prompt and no references.
        """
        instances = []

        # Create 100 instances for statistical validity
        # (DAT_GPT used 500, but 100 provides good balance for HELM)
        for i in range(100):
            instances.append(
                Instance(
                    input=Input(text=self.BASE_PROMPT),
                    references=[],  # No ground truth; evaluated by semantic distance metric
                    split=TEST_SPLIT,
                )
            )

        return instances
