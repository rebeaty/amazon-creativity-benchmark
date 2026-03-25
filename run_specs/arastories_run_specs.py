"""HELM Run Specs for arastories."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_FLUENCY = """\
Evaluate the FLUENCY of the generated Arabic story. Fluency measures how natural, grammatically correct, and readable the Arabic text is.

Score 1: Incomprehensible or severely broken Arabic language
Score 2: Frequent grammatical errors that impede understanding
Score 3: Generally understandable but with noticeable awkwardness
Score 4: Reads naturally with only minor linguistic issues
Score 5: Perfectly fluent, native-quality Arabic writing
"""

_RUBRIC_COHERENCE = """\
Evaluate the COHERENCE of the generated Arabic story. Coherence measures whether the story has logical flow, consistent characters, and a sensible narrative arc.

Score 1: No logical connection between sentences or ideas
Score 2: Major gaps in logic or contradictions in the narrative
Score 3: Mostly follows a thread but with some disconnects
Score 4: Well-structured narrative with minor inconsistencies
Score 5: Perfectly coherent story with clear, consistent narrative arc
"""

_RUBRIC_FOLLOWING_INSTRUCTIONS = """\
Evaluate how well the generated Arabic story FOLLOWS THE INSTRUCTIONS given in the prompt. Instructions specify age group, setting, ending type, character count, country, and other story constraints.

Score 1: Ignores nearly all specified constraints
Score 2: Addresses only 1-2 of the specified constraints
Score 3: Addresses most constraints with some notable omissions
Score 4: Addresses all constraints with only minor deviations
Score 5: Perfectly fulfills every specified constraint in the prompt
"""

_RUBRIC_CONSISTENCY = """\
Evaluate the CONSISTENCY of the generated Arabic story. Consistency measures whether characters, settings, and plot details remain stable throughout the story.

Score 1: Characters or settings change contradictorily without explanation
Score 2: Multiple inconsistencies that disrupt the reading experience
Score 3: Generally consistent with some minor continuity errors
Score 4: Consistent throughout with at most one minor inconsistency
Score 5: Perfectly consistent — every detail is stable and logical
"""

_RUBRIC_VARIETY = """\
Evaluate the VARIETY of the generated Arabic story. Variety measures the richness of vocabulary, sentence structures, and narrative techniques used.

Score 1: Monotonous, repetitive vocabulary and sentence structure throughout
Score 2: Limited variety; relies heavily on simple repeated patterns
Score 3: Some variety in expression but falls back on repetitive structures
Score 4: Good variety with diverse vocabulary and sentence constructions
Score 5: Rich, varied language with sophisticated narrative techniques
"""


@run_spec_function("arastories")
def get_arastories_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.arastories_scenario.AraStoriesScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,  # ASSUMPTION: zero-shot, no TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "fluency"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "coherence"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "following_instructions"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "consistency"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "variety"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-0125-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "fluency",
                "rubric": _RUBRIC_FLUENCY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-0125-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "coherence",
                "rubric": _RUBRIC_COHERENCE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-0125-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "following_instructions",
                "rubric": _RUBRIC_FOLLOWING_INSTRUCTIONS,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-0125-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "consistency",
                "rubric": _RUBRIC_CONSISTENCY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-0125-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "variety",
                "rubric": _RUBRIC_VARIETY,
            },
        ),
    ]

    return RunSpec(
        name="arastories",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "arastories"],
        annotators=annotators,
    )
