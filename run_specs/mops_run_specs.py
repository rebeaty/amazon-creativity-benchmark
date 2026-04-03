"""HELM Run Specs for mops."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_FASCINATION = """\
Evaluate how FASCINATING the generated movie premise or story concept is.
Consider how engaging, interesting, and captivating the premise would be to an audience.

Score 1: Premise is dull, boring, and completely unengaging
Score 2: Mostly uninteresting premise with minimal appeal
Score 3: Somewhat interesting premise with moderate audience appeal
Score 4: Genuinely fascinating premise that would engage most audiences
Score 5: Exceptionally fascinating premise with broad and compelling appeal
"""

_RUBRIC_LLM_JUDGE_ORIGINALITY = """\
Evaluate the ORIGINALITY of the generated movie premise or story concept.
Consider how novel, fresh, and unprecedented the concept is compared to existing films.

Score 1: Premise is a direct copy or cliche of existing films
Score 2: Mostly derivative with only minor original elements
Score 3: Some original elements mixed with familiar story patterns
Score 4: Notably original concept that stands out from existing films
Score 5: Highly original, unprecedented concept with exceptional creative novelty
"""


@run_spec_function("mops")
def get_mops_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.mops_scenario.MoPSPremiseScenario",
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
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicReferenceMetric", args={}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_fascination"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_originality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-turbo",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_fascination",
                "rubric": _RUBRIC_LLM_JUDGE_FASCINATION,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-turbo",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_originality",
                "rubric": _RUBRIC_LLM_JUDGE_ORIGINALITY,
            },
        ),
    ]

    return RunSpec(
        name="mops",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "mops"],
        annotators=annotators,
    )
