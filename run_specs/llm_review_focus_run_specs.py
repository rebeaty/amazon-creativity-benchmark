"""HELM Run Specs for llm_review_focus."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_QUALITY = """\
Evaluate the quality of the generated LLM review focus or research direction proposal.
Consider relevance, specificity, scientific rigor, and potential impact of the proposed focus.

Score 1: Proposal is vague, irrelevant, or scientifically unsound
Score 2: Poor quality proposal with major gaps in specificity or rigor
Score 3: Adequate proposal with a reasonably clear focus
Score 4: Good quality proposal with clear focus and scientific merit
Score 5: Excellent proposal with exceptional clarity, rigor, and potential scientific impact
"""


@run_spec_function("llm_review_focus")
def get_llm_review_focus_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.llm_review_focus_scenario.LLMReviewFocusScenario",
        args={},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",  # NOTE: scenario handles prompting internally
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=5,  # ASSUMPTION: few-shot, TRAIN_SPLIT seen
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_quality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "llm_judge_quality",
                "rubric": _RUBRIC_LLM_JUDGE_QUALITY,
            },
        ),
    ]

    return RunSpec(
        name="llm_review_focus",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "llm_review_focus"],
        annotators=annotators,
    )
