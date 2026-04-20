"""HELM Run Specs for grapheval_review_advisor."""

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
Evaluate the quality of the generated review advisory response.
Consider accuracy of advice, relevance to the review context, and overall helpfulness of the advisory content.

Score 1: Advisory is incorrect, irrelevant, or completely unhelpful
Score 2: Poor advice with major gaps in reasoning or relevance
Score 3: Adequate advisory meeting basic criteria
Score 4: Good advisory with sound and relevant guidance
Score 5: Excellent advisory demonstrating expert-level judgment and highly useful guidance
"""


@run_spec_function("grapheval_review_advisor")
def get_grapheval_review_advisor_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.grapheval_review_advisor_scenario.GraphEvalReviewAdvisorScenario",
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
        name="grapheval_review_advisor",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "grapheval_review_advisor"],
        annotators=annotators,
    )
