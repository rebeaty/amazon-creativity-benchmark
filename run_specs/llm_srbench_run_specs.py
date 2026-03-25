"""HELM Run Specs for llm_srbench."""

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
Evaluate the CORRECTNESS of the generated symbolic regression expression or solution.
Consider mathematical validity, fitness to the data pattern, and correctness of the formula.

Score 1: Expression is mathematically invalid or completely wrong
Score 2: Mostly incorrect with fundamental formula errors
Score 3: Partially correct expression with some valid mathematical structure
Score 4: Mostly correct expression with minor errors
Score 5: Fully correct expression that accurately captures the underlying data pattern
"""


@run_spec_function("llm_srbench")
def get_llm_srbench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.llm_srbench_scenario.LlmSrbenchScenario",
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
        MetricSpec(class_name="helm.benchmark.metrics.evaluate_reference_metrics.compute_reference_metrics", args={}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_correctness"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_correctness",
                "rubric": _RUBRIC_LLM_JUDGE_CORRECTNESS,
            },
        ),
    ]

    return RunSpec(
        name="llm_srbench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "llm_srbench"],
        annotators=annotators,
    )
