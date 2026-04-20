"""HELM Run Specs for creativemath."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_CORRECTNESS = """\
Evaluate the CORRECTNESS of the generated creative mathematical solution.
Consider mathematical validity, logical soundness, and whether the solution correctly solves the problem.

Score 1: Completely incorrect solution with fundamental mathematical errors
Score 2: Mostly incorrect with some valid steps but wrong conclusion
Score 3: Partially correct solution with some valid reasoning
Score 4: Mostly correct solution with minor errors or gaps
Score 5: Fully correct, mathematically rigorous and valid solution
"""


@run_spec_function("creativemath")
def get_creativemath_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.creativemath_scenario.CreativeMathScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_correctness"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "llm_judge_correctness",
                "rubric": _RUBRIC_LLM_JUDGE_CORRECTNESS,
            },
        ),
    ]

    return RunSpec(
        name="creativemath",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "creativemath"],
        annotators=annotators,
    )
