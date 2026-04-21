"""HELM Run Specs (SAMPLED MIRROR) for CS4.

Two evaluation units — one per dataset type — each with the reproducible
200-item sampler applied at the scenario level:
  - cs4_sampled_instruction
  - cs4_sampled_story

Adapter / annotator config mirrors the original [cs4_run_specs.py] so the
open-ended creative writing output + LLM-judge flow is identical.
"""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


_RUBRIC_LLM_JUDGE_CREATIVITY = """\
Evaluate the CREATIVITY of the generated content for a creative self-supervised story scenario.
Consider originality, narrative creativity, imaginative elements, and overall creative quality.

Score 1: No creative elements; formulaic and predictable content
Score 2: Minimal creativity; mostly conventional narrative choices
Score 3: Some creative elements mixed with conventional storytelling
Score 4: Notably creative content with strong original narrative elements
Score 5: Highly creative, imaginative content demonstrating exceptional storytelling creativity
"""


def _build_spec(dataset_type: str) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name="scenarios.cs4_sampled_scenario.CS4SampledScenario",
        args={"dataset_type": dataset_type},
    )

    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=512,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": "llm_judge_creativity"},
        ),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4o",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "llm_judge_creativity",
                "rubric": _RUBRIC_LLM_JUDGE_CREATIVITY,
            },
        ),
    ]

    return RunSpec(
        name=f"cs4_sampled:subtask={dataset_type}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "cs4_sampled"],
        annotators=annotators,
    )


@run_spec_function("cs4_sampled_instruction")
def get_cs4_sampled_instruction_spec() -> RunSpec:
    return _build_spec("instruction")


@run_spec_function("cs4_sampled_story")
def get_cs4_sampled_story_spec() -> RunSpec:
    return _build_spec("story")
