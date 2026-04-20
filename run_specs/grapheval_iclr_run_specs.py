"""HELM Run Specs for grapheval_iclr."""

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
Evaluate the quality of the ICLR review or paper assessment generated.
Consider accuracy of evaluation, depth of technical understanding, and quality of the review reasoning.

Score 1: Review is inaccurate, unhelpful, or shows no technical understanding
Score 2: Poor review with major gaps in technical reasoning
Score 3: Adequate review meeting basic ICLR review criteria
Score 4: Good technical review with sound reasoning and useful insights
Score 5: Excellent review demonstrating expert-level technical judgment and depth
"""


@run_spec_function("grapheval_iclr")
def get_grapheval_iclr_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.grapheval_iclr_scenario.GraphEvalICLRScenario",
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
        name="grapheval_iclr",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "grapheval_iclr"],
        annotators=annotators,
    )
