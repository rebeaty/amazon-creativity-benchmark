"""HELM Run Specs for eqbench_creative_writing_v3."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_ELO_RATING = """\
Compare the creative writing piece against a reference to determine relative quality.
Evaluate based on literary craft, originality, emotional resonance, and overall writing quality.

Score 1: The generated piece is substantially inferior in literary quality
Score 2: The generated piece is somewhat weaker in creative craft
Score 3: Both pieces are of roughly equal creative quality
Score 4: The generated piece shows stronger creative craft and quality
Score 5: The generated piece is substantially superior in literary excellence
"""

_RUBRIC_RUBRIC_SCORE = """\
Evaluate the creative writing piece according to the detailed EQBench rubric criteria.
Consider creativity, emotional depth, narrative craft, character, and overall literary merit.

Score 1: Fails on nearly all rubric criteria
Score 2: Meets only minimal rubric criteria
Score 3: Meets most rubric criteria adequately
Score 4: Meets all rubric criteria well
Score 5: Exceeds all rubric criteria with exceptional literary quality
"""


@run_spec_function("eqbench_creative_writing_v3")
def get_eqbench_creative_writing_v3_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.eqbench_creative_writing_v3_scenario.EQBenchCreativeWritingV3Scenario",
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
        num_outputs=3,
        max_tokens=2048,
        temperature=0.7,
        stop_sequences=[],
    )

    metric_specs = [
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "elo_rating"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "rubric_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "anthropic/claude-sonnet-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 1024,
                "metric_name": "elo_rating",
                "rubric": _RUBRIC_ELO_RATING,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "anthropic/claude-sonnet-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 512,
                "metric_name": "rubric_score",
                "rubric": _RUBRIC_RUBRIC_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="eqbench_creative_writing_v3",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "eqbench_creative_writing_v3"],
        annotators=annotators,
    )
