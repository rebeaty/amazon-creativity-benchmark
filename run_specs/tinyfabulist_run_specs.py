"""HELM Run Specs for tinyfabulist."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_GRAMMAR_SCORE = """\
Evaluate the GRAMMAR AND STYLE of the generated fable.
Consider grammatical correctness, writing clarity, and appropriateness of style for a fable.

Score 1-3: Significant grammatical errors that impede understanding
Score 4-6: Some errors but generally readable
Score 7-10: Clean, polished writing with appropriate language and style for a fable
"""

_RUBRIC_CREATIVITY_SCORE = """\
Evaluate the CREATIVITY AND ORIGINALITY of the generated fable.
Consider how fresh, innovative, and original the approach is while maintaining classic fable structure.

Score 1-3: Derivative, predictable, or clichéd content
Score 4-6: Contains some original elements but follows familiar patterns
Score 7-10: Fresh perspective, innovative approach while maintaining classic fable structure
"""


@run_spec_function("tinyfabulist")
def get_tinyfabulist_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.tinyfabulist_scenario.TinyFabulistScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "grammar_score"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "creativity_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/o3-mini-2025-01-31",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 350,
                "metric_name": "grammar_score",
                "rubric": _RUBRIC_GRAMMAR_SCORE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/o3-mini-2025-01-31",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 350,
                "metric_name": "creativity_score",
                "rubric": _RUBRIC_CREATIVITY_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="tinyfabulist",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "tinyfabulist"],
        annotators=annotators,
    )
