"""HELM Run Specs for speak_to_structure."""

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
Evaluate the CORRECTNESS of the generated structured representation from natural language.
Consider whether the structure correctly captures the meaning and relationships in the input.

Score 1: Structure is completely incorrect or invalid
Score 2: Mostly incorrect with major structural errors
Score 3: Partially correct with some valid structural elements
Score 4: Mostly correct structure with minor errors
Score 5: Fully correct structure that perfectly captures the input meaning
"""


@run_spec_function("speak_to_structure")
def get_speak_to_structure_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.speak_to_structure_scenario.SpeakToStructureScenario",
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
        MetricSpec(class_name="metrics.validity_metric.ValidityMetric", args={}),
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
        name="speak_to_structure",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "speak_to_structure"],
        annotators=annotators,
    )
