"""HELM Run Specs for gauss."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_RUBRIC_SCORE = """\
Evaluate the mathematical solution using the provided problem-specific rubric.
Assign points based on correctness of approach, mathematical reasoning, and final answer.

Score 0: Solution shows no understanding or is completely wrong
Score 1-2: Solution shows minimal understanding with fundamental errors
Score 3-4: Solution shows partial understanding with some correct steps
Score 5-6: Solution is mostly correct with minor errors
Score 7+: Solution is fully correct demonstrating complete mathematical mastery
"""


@run_spec_function("gauss")
def get_gauss_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.gauss_scenario.GAUSSScenario",
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
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "rubric_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 64,
                "metric_name": "rubric_score",
                "rubric": _RUBRIC_RUBRIC_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="gauss",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "gauss"],
        annotators=annotators,
    )
