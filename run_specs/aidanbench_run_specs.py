"""HELM Run Specs for aidanbench."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_COHERENCE_SCORE = """\
Evaluate the COHERENCE, quality, and appropriateness of the response to an open-ended question.
A score of 0 means completely incoherent or off-topic. A score of 100 means a perfect, insightful, well-reasoned response.

Score 1: Completely incoherent, irrelevant, or offensive response
Score 25: Vaguely related but poorly argued and low quality
Score 50: Adequate response that addresses the question at a surface level
Score 75: Good quality, coherent response with solid reasoning
Score 100: Exceptional, insightful, perfectly reasoned and articulated response
"""


@run_spec_function("aidanbench")
def get_aidanbench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.aidanbench_scenario.AidanBenchScenario",
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
        num_outputs=30,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="helm.benchmark.metrics.disinformation_metrics.DisinformationMetric", args={"name": "self_bleu"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "coherence_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/o1-mini",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 64,
                "metric_name": "coherence_score",
                "rubric": _RUBRIC_COHERENCE_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="aidanbench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "aidanbench"],
        annotators=annotators,
    )
