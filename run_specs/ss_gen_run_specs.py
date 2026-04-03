"""HELM Run Specs for ss_gen."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_COHERENCE = """\
Evaluate the COHERENCE of the generated story segment or continuation.
Consider logical flow, narrative consistency, and how well it connects to the preceding context.

Score 1: Generated segment is completely incoherent with the story context
Score 2: Poor coherence with major narrative inconsistencies
Score 3: Adequate coherence with some narrative flow issues
Score 4: Good coherence that maintains narrative consistency
Score 5: Excellent coherence with perfect narrative flow and consistency
"""


@run_spec_function("ss_gen")
def get_ss_gen_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.ss_gen_scenario.SSGenScenario",
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
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicReferenceMetric", args={}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_coherence"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_coherence",
                "rubric": _RUBRIC_LLM_JUDGE_COHERENCE,
            },
        ),
    ]

    return RunSpec(
        name="ss_gen",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "ss_gen"],
        annotators=annotators,
    )
