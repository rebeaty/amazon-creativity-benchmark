"""
Central registry for all Amazon Creativity Benchmark HELM scenarios.

Imports every Scenario subclass and exposes them via ``__all__`` and
``SCENARIO_REGISTRY`` so HELM's module discovery and RunSpec builders
can locate all scenarios from a single import path:

    from scenarios import SCENARIO_REGISTRY
    scenario_cls = SCENARIO_REGISTRY["hummus"]
    scenario = scenario_cls(subset="classification")

Special cases handled here
--------------------------
* ``graphragbench-wrongone/``  — hyphen in directory name is illegal for
  Python imports; loaded via ``importlib.util`` and aliased as
  ``GraphRAGBenchWrongoneScenario``.
* ``vflute/`` vs ``v_flute/``  — both define a class called
  ``VFluteScenario`` but cover different tasks; the ``v_flute`` version
  (subset-parameterised, name="v_flute") is re-exported as
  ``VFluteSubsetScenario`` to avoid the name collision.
* ``material_generation_benchmark/`` — contains three separate
  ``*_scenario.py`` files instead of a single ``scenario.py``.
* ``fscg8_scenario.py`` / ``kiva_scenario.py`` — loose top-level files
  whose companion subdirectories contain no ``scenario.py``.
* ``recombination_extraction_scenario.py`` — top-level duplicate of
  ``recombination_extraction/scenario.py``; only the subdirectory
  version is imported here.
"""

import importlib.util
import os
from typing import Dict, Type

from helm.benchmark.scenarios.scenario import Scenario

# ---------------------------------------------------------------------------
# Internal helper — load a scenario module from an absolute file path.
# Needed for directories whose names contain characters (e.g., hyphens)
# that are illegal in Python identifiers.
# ---------------------------------------------------------------------------

def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_HERE = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Standard scenarios — one Scenario subclass per subdirectory
# ---------------------------------------------------------------------------

from .aaar.scenario import AaarScenario
from .aidanbench.scenario import AidanBenchScenario
from .amuse_chord_generation.scenario import AmuseChordGenerationScenario
from .analobench.scenario import AnalobenchScenario
from .arastories.scenario import AraStoriesScenario
from .arena_hard_creative.scenario import ArenaHardCreativeScenario
from .arn.scenario import ARNScenario
from .artinsight.scenario import ArtInsightScenario
from .assocam.scenario import AssoCiAmScenario
from .ava.scenario import AVAScenario
from .balancecc_prompt_generation.scenario import BalanceCCPromptGenerationScenario
from .balderdash.scenario import BalderdashScenario
from .banner_request_400.scenario import BannerRequest400Scenario
from .bhp_hypothesis_generation.scenario import BHPHypothesisGenerationScenario
from .brainteaser.scenario import BrainteaserScenario
from .c3_crosstalk.scenario import C3CrosstalkScenario
from .calligrapher.scenario import CalligrapherScenario
from .cdat.scenario import CDATScenario
from .chinese_homophonic_puns.scenario import ChineseHomophonicPunsScenario
from .clef_joker_2025_task2.scenario import CLEFJoker2025Task2Scenario
from .cm3d.scenario import CM3DScenario
from .conceptual_design.scenario import ConceptualDesignScenario
from .convbench.scenario import ConvBenchScenario
from .cpers.scenario import CPersScenario
from .creation_mmbench.scenario import CreationMMBenchScenario
from .creative_pair.scenario import CreativePairScenario
from .creative_process.scenario import CreativeProcessScenario
from .creativemath.scenario import CreativeMathScenario
from .creatset.scenario import CreataSetScenario
from .critics_story.scenario import CriticsStoryScenario
from .crowd_vote.scenario import CrowdVoteScenario
from .crowdcounter.scenario import CrowdCounterScenario
from .cs4.scenario import CS4Scenario
from .csd100.scenario import CSD100Scenario
# d_humor exposes three classes in a single file
from .d_humor.scenario import (
    DHumorDetectionScenario,
    DHumorTargetScenario,
    DHumorIntensityScenario,
)
from .dat.scenario import DATScenario
from .dat_creative_writing.scenario import DATCreativeWritingScenario
from .deep_math.scenario import DeepMathCreativeScenario
from .discovery_bench.scenario import DiscoveryBenchScenario
from .diverse_not_short.scenario import DiverseNotShortScenario
from .dpt.scenario import DptScenario
from .esp_dataset.scenario import ESPDatasetScenario
from .fann_or_flop.scenario import FannOrFlopScenario
from .fig_qa.scenario import FigQAScenario
from .flute_filtered.scenario import FLUTEFilteredScenario
from .funqa.scenario import FunQAScenario
from .future_ideas.scenario import FutureIdeasScenario
from .futuregen.scenario import FuturegenScenario
from .fuxibench.scenario import FuxiBenchScenario
from .geo_story.scenario import GeoStoryScenario
from .grapheval_ai_researcher.scenario import GraphEvalAIResearcherScenario
from .grapheval_iclr.scenario import GraphEvalICLRScenario
from .grapheval_review_advisor.scenario import GraphEvalReviewAdvisorScenario
from .graphrag_bench.scenario import GraphRAGBenchScenario
from .hummus.scenario import HummusScenario
from .humor_transfer.scenario import HumorTransferScenario
from .hypobench.scenario import HypoBenchScenario
from .hypogen.scenario import HypoGenScenario
from .idrbench.scenario import IDRBenchScenario
from .ii_bench.scenario import IIBenchScenario
from .irfl.scenario import IRFLScenario
from .javanese_sundanese_story_cloze.scenario import JavaneseSundaneseStoryClozeScenario
from .layoutsam_eval.scenario import LayoutSAMEvalScenario
from .lcc_metaphor.scenario import LCCMetaphorScenario
from .litbench.scenario import LitBenchScenario
from .liveideabench.scenario import LiveIdeaBenchScenario
from .llm_review_focus.scenario import LLMReviewFocusScenario
from .llm_srbench.scenario import LlmSrbenchScenario
from .macgyver.scenario import MacgyverScenario
from .mars.scenario import MARSScenario
from .mixassist.scenario import MixAssistScenario
from .met_meme.scenario import METMemeScenario
from .meta4xnli.scenario import Meta4XNLIScenario
from .mineanybuild.scenario import MineAnyBuildScenario
from .miqa.scenario import MiQAScenario
from .moh_x.scenario import MOHXScenario
from .mops.scenario import MoPSPremiseScenario
from .munch.scenario import MUNCHScenario
from .muse_perception.scenario import MuSePerceptionScenario
from .neocoder.scenario import NeocoderScenario
from .newyorker_humor.scenario import NewYorkerHumorScenario
from .nyt_connections.scenario import NYTConnectionsScenario
from .ocw.scenario import OnlyConnectWallScenario
from .ocw_connections.scenario import OnlyConnectConnectionsScenario
from .oogiri_go.scenario import OogiriGOScenario
from .pace.scenario import PACEScenario
from .permpst.scenario import PerMPSTScenario
from .poetmt.scenario import PoetMTScenario
from .pollux_creativity.scenario import POLLUXCreativityScenario
from .pron_vs_prompt.scenario import PronVsPromptScenario
from .proparalogy.scenario import ProparaLogyScenario
from .protein_bench.scenario import ProteinBenchScenario
from .pun_eval.scenario import PunEvalScenario
from .pun2pun.scenario import Pun2PunScenario
from .puzzleworld.scenario import PuzzleWorldScenario
# recombination_extraction: prefer subdirectory over top-level duplicate
from .recombination_extraction.scenario import RecombinationExtractionScenario
from .research_idea_execution.scenario import ResearchIdeaExecutionScenario
from .riddlesense.scenario import RiddlesenseScenario
from .robotoolbench.scenario import RoboToolBenchScenario
from .rpgbench.scenario import RpgBenchScenario
from .scar.scenario import SCARScenario
from .science_analogies.scenario import ScienceAnalogiesScenario
from .scimon.scenario import SciMONScenario
from .sdat.scenario import SDATScenario
from .showerthoughts.scenario import ShowerthoughtsScenario
from .simile_generation.scenario import SimileGenerationScenario
from .slang_generation.scenario import SlangGenerationScenario
from .sonnet_or_not_bot.scenario import SonnetOrNotBotScenario
from .speak_to_structure.scenario import SpeakToStructureScenario
from .splat.scenario import SPLATScenario
from .story_generation_rocstories.scenario import StoryGenerationScenario
from .story_quality.scenario import StoryQualityScenario
from .storyer.scenario import StoryERScenario
from .sudoku_bench.scenario import SudokuBenchScenario
from .textlogo3k.scenario import TextLogo3KScenario
from .thenextchapter.scenario import TheNextChapterScenario
from .tiger_bench.scenario import TIGeRBenchScenario
from .tinyfabulist.scenario import TinyFabulistScenario
from .ttcw.scenario import TTCWScenario
from .unfun_corpus.scenario import UnfunCorpusScenario
# vflute: two independent implementations of the same paper.
# VFluteScenario        — vflute/  (name="vflute",  single-task)
# VFluteSubsetScenario  — v_flute/ (name="v_flute", subset-parameterised)
from .vflute.scenario import VFluteScenario
from .v_flute.scenario import VFluteScenario as VFluteSubsetScenario
from .vgsg.scenario import VGSGScenario
from .vietnamese_poem.scenario import VietnamesePoemScenario
from .webnovelbench.scenario import WebNovelBenchScenario
from .writingbench.scenario import WritingBenchScenario
from .yesbut_v2.scenario import YesButV2Scenario

# ---------------------------------------------------------------------------
# material_generation_benchmark — three separate scenario files, no scenario.py
# ---------------------------------------------------------------------------

from .material_generation_benchmark.carbon24_scenario import (
    MaterialGenerationCarbon24Scenario,
)
from .material_generation_benchmark.mp20_scenario import (
    MaterialGenerationMP20Scenario,
)
from .material_generation_benchmark.perov5_scenario import (
    MaterialGenerationPerov5Scenario,
)

# ---------------------------------------------------------------------------
# Loose top-level scenario files (no matching subdirectory scenario.py)
# ---------------------------------------------------------------------------

from .fscg8_scenario import FSCG8Scenario
from .kiva_scenario import KiVAScenario

# ---------------------------------------------------------------------------
# graphragbench-wrongone — hyphen in directory name requires importlib
# Aliased as GraphRAGBenchWrongoneScenario to avoid collision with
# graphrag_bench.GraphRAGBenchScenario imported above.
# ---------------------------------------------------------------------------

_wrongone_mod = _load_module_from_path(
    "graphragbench_wrongone_scenario",
    os.path.join(_HERE, "graphragbench-wrongone", "scenario.py"),
)
GraphRAGBenchWrongoneScenario: Type[Scenario] = _wrongone_mod.GraphRAGBenchScenario

# ---------------------------------------------------------------------------
# SCENARIO_REGISTRY
# Maps each scenario's ``name`` class attribute to its class.
# Used by RunSpec builders to resolve scenario names at runtime.
# When two classes share the same name attribute, the entry produced last
# in the list wins; duplicates are flagged in a comment below.
# ---------------------------------------------------------------------------

_ALL_CLASSES: list = [
    # A
    AaarScenario,
    AidanBenchScenario,
    AmuseChordGenerationScenario,
    AnalobenchScenario,
    AraStoriesScenario,
    ArenaHardCreativeScenario,
    ARNScenario,
    ArtInsightScenario,
    AssoCiAmScenario,
    AVAScenario,
    # B
    BalanceCCPromptGenerationScenario,
    BalderdashScenario,
    BannerRequest400Scenario,
    BHPHypothesisGenerationScenario,
    BrainteaserScenario,
    # C
    C3CrosstalkScenario,
    CalligrapherScenario,
    CDATScenario,
    ChineseHomophonicPunsScenario,
    CLEFJoker2025Task2Scenario,
    CM3DScenario,
    ConceptualDesignScenario,
    ConvBenchScenario,
    CPersScenario,
    CreationMMBenchScenario,
    CreativePairScenario,
    CreativeProcessScenario,
    CreativeMathScenario,
    CreataSetScenario,
    CriticsStoryScenario,
    CrowdVoteScenario,
    CrowdCounterScenario,
    CS4Scenario,
    CSD100Scenario,
    # D — d_humor has 3 classes
    DHumorDetectionScenario,
    DHumorTargetScenario,
    DHumorIntensityScenario,
    DATScenario,
    DATCreativeWritingScenario,
    DeepMathCreativeScenario,
    DiscoveryBenchScenario,
    DiverseNotShortScenario,
    DptScenario,
    # E
    ESPDatasetScenario,
    # F
    FannOrFlopScenario,
    FigQAScenario,
    FLUTEFilteredScenario,
    FSCG8Scenario,
    FunQAScenario,
    FutureIdeasScenario,
    FuturegenScenario,
    FuxiBenchScenario,
    # G
    GeoStoryScenario,
    GraphEvalAIResearcherScenario,
    GraphEvalICLRScenario,
    GraphEvalReviewAdvisorScenario,
    GraphRAGBenchScenario,
    GraphRAGBenchWrongoneScenario,  # may share name with GraphRAGBenchScenario
    # H
    HummusScenario,
    HumorTransferScenario,
    HypoBenchScenario,
    HypoGenScenario,
    # I
    IDRBenchScenario,
    IIBenchScenario,
    IRFLScenario,
    # J
    JavaneseSundaneseStoryClozeScenario,
    # K
    KiVAScenario,
    # L
    LayoutSAMEvalScenario,
    LCCMetaphorScenario,
    LitBenchScenario,
    LiveIdeaBenchScenario,
    LLMReviewFocusScenario,
    LlmSrbenchScenario,
    # M
    MacgyverScenario,
    MARSScenario,
    MaterialGenerationCarbon24Scenario,
    MaterialGenerationMP20Scenario,
    MaterialGenerationPerov5Scenario,
    METMemeScenario,
    Meta4XNLIScenario,
    MineAnyBuildScenario,
    MiQAScenario,
    MixAssistScenario,
    MOHXScenario,
    MoPSPremiseScenario,
    MUNCHScenario,
    MuSePerceptionScenario,
    # N
    NeocoderScenario,
    NewYorkerHumorScenario,
    NYTConnectionsScenario,
    # O
    OnlyConnectWallScenario,
    OnlyConnectConnectionsScenario,
    OogiriGOScenario,
    # P
    PACEScenario,
    PerMPSTScenario,
    PoetMTScenario,
    POLLUXCreativityScenario,
    PronVsPromptScenario,
    ProparaLogyScenario,
    ProteinBenchScenario,
    PunEvalScenario,
    Pun2PunScenario,
    PuzzleWorldScenario,
    # R
    RecombinationExtractionScenario,
    ResearchIdeaExecutionScenario,
    RiddlesenseScenario,
    RoboToolBenchScenario,
    RpgBenchScenario,
    # S
    SCARScenario,
    ScienceAnalogiesScenario,
    SciMONScenario,
    SDATScenario,
    ShowerthoughtsScenario,
    SimileGenerationScenario,
    SlangGenerationScenario,
    SonnetOrNotBotScenario,
    SpeakToStructureScenario,
    SPLATScenario,
    StoryGenerationScenario,
    StoryQualityScenario,
    StoryERScenario,
    SudokuBenchScenario,
    # T
    TextLogo3KScenario,
    TheNextChapterScenario,
    TIGeRBenchScenario,
    TinyFabulistScenario,
    TTCWScenario,
    # U
    UnfunCorpusScenario,
    # V
    VFluteScenario,        # name="vflute"
    VFluteSubsetScenario,  # name="v_flute"
    VGSGScenario,
    VietnamesePoemScenario,
    # W
    WebNovelBenchScenario,
    WritingBenchScenario,
    # Y
    YesButV2Scenario,
]

SCENARIO_REGISTRY: Dict[str, Type[Scenario]] = {
    cls.name: cls for cls in _ALL_CLASSES
}

# ---------------------------------------------------------------------------
# __all__
# Derived from globals() so aliased names (VFluteSubsetScenario,
# GraphRAGBenchWrongoneScenario) are exported correctly regardless of the
# class's internal __name__ attribute.
# ---------------------------------------------------------------------------

__all__ = sorted(
    name
    for name, obj in globals().items()
    if (
        isinstance(obj, type)
        and issubclass(obj, Scenario)
        and obj is not Scenario
        and not name.startswith("_")
    )
) + ["SCENARIO_REGISTRY"]
