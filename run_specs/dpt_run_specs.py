"""HELM Run Specs for dpt."""

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
Evaluate the CREATIVITY of the generated response to a divergent production task.
Consider the range, originality, and unexpectedness of ideas produced.

Score 1: Only obvious, common, and predictable responses
Score 2: Mostly conventional with only slight variations
Score 3: Mix of conventional and somewhat original responses
Score 4: Notably creative responses with original and unexpected ideas
Score 5: Highly creative, diverse, and original responses demonstrating exceptional divergent thinking
"""


@run_spec_function("dpt")
def get_dpt_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.dpt_scenario.DptScenario",
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
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_creativity",
                "rubric": _RUBRIC_LLM_JUDGE_CREATIVITY,
            },
        ),
    ]

    return RunSpec(
        name="dpt",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "dpt"],
        annotators=annotators,
    )
