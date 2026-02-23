"""
HELM Scenario: LLM-MA Balderdash — Fake Definition Generation

Paper: "Evaluating Creativity and Deception in Large Language Models:
        The Balderdash Game" (arXiv:2411.10422, Nov 2024)
Authors: Parsa Hejabi et al.
Code:    https://github.com/ParsaHejabi/Simulation-Framework-for-Multi-Agent-Balderdash

Task: Given an obscure English word, generate a convincing but false dictionary
definition — one that sounds plausible enough to fool readers into believing
it is genuine. Tests creative deception: linguistic creativity, dictionary-style
writing, and plausible confabulation.

Based on the Balderdash board game:
  - An obscure word is revealed; players secretly write fake definitions
  - Players try to fool others into picking their fake over the real definition
  - Scoring rewards convincing fakes and correctly identifying the real one

This scenario focuses on the generation subtask: given a word (no real
definition shown), produce a fake definition that sounds authentic.

Real definitions are embedded per-word for judge calibration ONLY — the model
never sees the real definition in its prompt. This faithfully replicates the
game mechanic where players must invent without knowing the true meaning.

Prompt (adapted from paper arXiv:2411.10422, game task description):
  "You are playing Balderdash, a word game where you must write a fake but
   convincing dictionary definition for an obscure word. The definition must
   sound authentic and plausible — like it could appear in a real dictionary.
   Do not use the word itself in your definition.

   Word: {WORD}

   Write a single fake dictionary definition (1–2 sentences, dictionary style)."

Dataset: 50 curated obscure English words, drawn from the Wordnik Balderdash
  collection and classic word game vocabulary. Embedded in-code; no download
  required. Real definitions included for judge calibration (not shown to model).

Fields used:   word (curated list, embedded in-code)
Fields skipped: real_definition (embedded for judge calibration only)
Prompt source: Adapted from paper Section 3 (game task specification)
Evaluation: llm_judge (see annotator_notes.md)
  Dimensions: convincingness, plausibility, dictionary_style, originality (1–5)
  Judge compares fake vs. real definition to check for unintentional accuracy.

Parameters:
  domain: "obscure" | "common" | "all" (default: "obscure")
    obscure: 50 Balderdash-style rare words (core benchmark)
    common:  20 frequent English words (control condition from paper)
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

# ---------------------------------------------------------------------------
# Word lists — real definitions embedded for judge calibration ONLY
# Model is never shown the real definition in its prompt
# ---------------------------------------------------------------------------

# 50 obscure words (Balderdash-style) + real definitions for judge use
# Sources: Wordnik Balderdash collection, Merriam-Webster, Wiktionary
_OBSCURE_WORDS = [
    ("zugzwang",        "a situation in chess in which a player is forced to make a move that worsens their position"),
    ("omphaloskepsis",  "contemplation of one's navel as part of a mystical exercise"),
    ("widdershins",     "in a direction contrary to the apparent course of the sun; counterclockwise"),
    ("octothorpe",      "the symbol #, used in telecommunications and social media"),
    ("callipygian",     "having well-shaped or finely developed buttocks"),
    ("borborygmus",     "a rumbling or gurgling noise made by the movement of fluid and gas in the intestines"),
    ("petrichor",       "a pleasant, distinctive smell frequently accompanying the first rain after a long dry spell"),
    ("apricity",        "the warmth of the sun in winter"),
    ("vellichor",       "the strange wistfulness of used bookshops"),
    ("hiraeth",         "a Welsh word for a homesickness tinged with grief for something lost, or the longing for Wales"),
    ("bloviate",        "to talk at length, especially in an inflated or empty way"),
    ("flibbertigibbet", "a frivolous, flighty, or excessively talkative person"),
    ("absquatulate",    "to leave abruptly"),
    ("cattywampus",     "in disarray; askew"),
    ("lagniappe",       "something given as a bonus or extra gift"),
    ("katzenjammer",    "a confused, chaotic state or uproar; a massive hangover"),
    ("snollygoster",    "a shrewd, unprincipled person, especially a politician"),
    ("taradiddle",      "a petty lie; pretentious nonsense"),
    ("scripturient",    "having a consuming passion to write"),
    ("lucubrate",       "to write or study, especially by night"),
    ("titivate",        "to make small enhancements to one's appearance; to spruce up"),
    ("foofaraw",        "a great fuss or disturbance about something very insignificant"),
    ("logorrhea",       "a tendency to extreme loquaciousness; excessive and often incoherent talkativeness"),
    ("hebetude",        "the state of being dull or lethargic"),
    ("bumfuzzle",       "to confuse or perplex someone"),
    ("callithump",      "a boisterous and discordant band or parade"),
    ("wamble",          "to feel nauseous; to move unsteadily"),
    ("zymurgy",         "the branch of applied chemistry dealing with fermentation processes, as in brewing"),
    ("gallimaufry",     "a confused jumble or medley of things"),
    ("palimpsest",      "a manuscript or piece of writing material on which writing has been effaced to make room for later writing but of which traces remain"),
    ("objurgation",     "a harsh rebuke or criticism"),
    ("tergiversation",  "the making of contradictory or evasive statements; the desertion of a cause or party"),
    ("quodlibet",       "a philosophical or theological point proposed for debate; a lighthearted medley of tunes"),
    ("funambulist",     "a tightrope walker"),
    ("obambulate",      "to walk about; to wander"),
    ("xenodochium",     "a home for strangers and pilgrims; an inn"),
    ("ergophobia",      "an abnormal and persistent fear of work or finding employment"),
    ("munificent",      "characterized by or displaying great generosity"),
    ("nescient",        "lacking knowledge; ignorant"),
    ("peristalsis",     "the involuntary constriction and relaxation of the muscles of the intestine, creating wave-like movements that push the contents of the canal forward"),
    ("kerfuffle",       "a commotion or fuss, especially one caused by conflicting views"),
    ("lollygag",        "to spend time aimlessly; to dawdle"),
    ("yaffle",          "the green woodpecker"),
    ("whiffet",         "a small, young, or unimportant person"),
    ("farrago",         "a confused mixture; a hotchpotch"),
    ("defenestration",  "the action of throwing someone out of a window"),
    ("sialoquent",      "one who sprays saliva when speaking"),
    ("xertz",           "to gulp down quickly and greedily"),
    ("jentacular",      "of or pertaining to breakfast"),
    ("vigesimation",    "the act of putting every twentieth person to death"),
]

# 20 common English words (control condition — paper §4.2)
# Oxford 3000 words; these are familiar, so models know real definitions
_COMMON_WORDS = [
    ("bridge",      "a structure carrying a road, path, railway, etc. across a river, road, or other obstacle"),
    ("cabinet",     "a cupboard with drawers or shelves for storing or displaying articles"),
    ("canvas",      "a strong, coarse unbleached cloth used to make sails, tents, awnings, and as a surface for oil painting"),
    ("chamber",     "a large room used for formal or public events"),
    ("crane",       "a large, tall machine used for moving heavy objects by suspending them from a projecting arm"),
    ("draft",       "a preliminary version of a piece of writing"),
    ("eagle",       "a large bird of prey with a massive hooked bill and long broad wings"),
    ("gallery",     "a room or building for the display or sale of works of art"),
    ("harbor",      "a place on the coast where ships may moor in shelter, especially one protected from rough water"),
    ("journal",     "a newspaper or magazine that deals with a particular subject or professional activity"),
    ("kernel",      "a softer, usually edible part of a nut, seed, or fruit stone contained within its hard shell"),
    ("lance",       "a long weapon with a wooden shaft and a pointed steel head, formerly used by a mounted knight"),
    ("margin",      "the edge or border of something"),
    ("novel",       "a fictitious prose narrative of book length, typically representing character and action with some degree of realism"),
    ("orbit",       "the curved path of a celestial object or spacecraft around a star, planet, or moon"),
    ("panel",       "a flat or curved component that forms or is set into the surface of a door, wall, or ceiling"),
    ("quarter",     "each of four equal or corresponding parts into which something is or can be divided"),
    ("register",    "an official list or record of names or items"),
    ("scale",       "a device for weighing; also, one of the small overlapping plates protecting the skin of fish"),
    ("vault",       "a roof in the form of an arch or a series of arches, typical of churches and similar buildings"),
]

_DOMAIN_MAP = {
    "obscure": _OBSCURE_WORDS,
    "common":  _COMMON_WORDS,
}

_VALID_DOMAINS = ["obscure", "common", "all"]

# Prompt adapted from paper Section 3 game task specification
_INSTRUCTION = (
    "You are playing Balderdash, a word game where you must write a fake but "
    "convincing dictionary definition for an obscure word. The definition must "
    "sound authentic and plausible — as if it could appear in a real dictionary. "
    "Do not use the word itself in your definition. Do not say the definition is "
    "fake. Write in a formal, dictionary-entry style.\n\n"
    "Word: {word}\n\n"
    "Write a single fake dictionary definition (1–2 sentences)."
)


class BalderdashScenario(Scenario):
    """
    LLM-MA Balderdash — creative fake dictionary definition generation.

    Given an obscure word (no real definition shown), the model generates a
    convincing but false dictionary entry. Tests linguistic creativity,
    plausible confabulation, and dictionary-style writing.

    50 obscure words (core) + 20 common words (control condition). Real
    definitions embedded in-code for judge calibration only. No download needed.

    Parameters:
      domain: "obscure" (50 Balderdash words) | "common" (20 Oxford words)
              | "all" (70 total)
    """

    name = "balderdash"
    description = "github.com/ParsaHejabi/Simulation-Framework-for-Multi-Agent-Balderdash (arXiv:2411.10422)"
    tags = ["creativity", "language", "deception", "open_ended_generation", "word_games"]

    def __init__(self, domain: str = "obscure"):
        super().__init__()
        if domain not in _VALID_DOMAINS:
            raise ValueError(
                f"Unknown domain: {domain!r}. Must be one of {_VALID_DOMAINS}"
            )
        self.domain = domain

    def get_instances(self, output_path: str) -> List[Instance]:
        if self.domain == "all":
            word_pairs = _OBSCURE_WORDS + _COMMON_WORDS
        else:
            word_pairs = _DOMAIN_MAP[self.domain]

        instances = []
        for word, _real_definition in word_pairs:
            # Real definition intentionally excluded from prompt —
            # model must invent without knowing the true meaning (game rule).
            # _real_definition is retained here for judge calibration use only.
            prompt = _INSTRUCTION.format(word=word)

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],   # No correct answer; LLM-as-judge
                    split=TEST_SPLIT,
                )
            )

        return instances  # 50 (obscure) | 20 (common) | 70 (all)
