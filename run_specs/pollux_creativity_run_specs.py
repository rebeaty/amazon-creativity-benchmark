"""HELM Run Specs for pollux_creativity."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_CREATIVITY = """\
Evaluate the CREATIVITY of the generated response in this creative generation task.
Consider originality, divergent thinking, imaginative expression, and creative quality.

Score 1: Response is completely uncreative and formulaic
Score 2: Minimal creativity with mostly conventional expression
Score 3: Some creative elements with moderate originality
Score 4: Genuinely creative response with strong original elements
Score 5: Highly creative response demonstrating exceptional originality and imaginative thinking
"""

_RUBRIC_LLM_JUDGE_ORIGINALITY = """\
Evaluate the ORIGINALITY of the generated response in this creative task.
Consider how unique, novel, and unprecedented the ideas are compared to conventional responses.

Score 1: Response contains no original ideas; completely derivative
Score 2: Slightly varied but largely unoriginal
Score 3: Some original elements within mostly conventional thinking
Score 4: Notably original with ideas that clearly stand out
Score 5: Exceptionally original with ideas that are novel and unprecedented
"""


@run_spec_function("pollux_creativity")
def get_pollux_creativity_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.pollux_creativity_scenario.POLLUXCreativityScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_creativity"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_originality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "llm_judge_creativity",
                "rubric": _RUBRIC_LLM_JUDGE_CREATIVITY,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "llm_judge_originality",
                "rubric": _RUBRIC_LLM_JUDGE_ORIGINALITY,
            },
        ),
    ]

    return RunSpec(
        name="pollux_creativity",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "pollux_creativity"],
        annotators=annotators,
    )
