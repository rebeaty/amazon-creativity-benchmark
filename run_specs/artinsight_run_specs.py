"""HELM Run Specs for artinsight."""

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
Evaluate the quality of the art analysis or interpretation based on a rubric.
Consider depth of insight, accuracy of observation, and quality of reasoning about the artwork.

Score 1: Superficial or incorrect analysis with no meaningful insight
Score 2: Minimal insight with significant gaps in understanding
Score 3: Adequate analysis covering basic elements of the artwork
Score 4: Thoughtful analysis showing good art knowledge and interpretation
Score 5: Exceptional insight demonstrating deep understanding and original perspective
"""


@run_spec_function("artinsight")
def get_artinsight_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.artinsight_scenario.ArtInsightScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "rubric_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4o",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "rubric_score",
                "rubric": _RUBRIC_RUBRIC_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="artinsight",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "artinsight"],
        annotators=annotators,
    )
