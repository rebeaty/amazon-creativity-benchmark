"""HELM Run Specs for arena_hard_v01."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_WIN_RATE = """\
Evaluate the quality of the response in a head-to-head comparison with a reference answer.
Consider helpfulness, accuracy, depth of understanding, and overall quality.

Score 1: Generated response is significantly worse than reference
Score 2: Generated response is somewhat worse than reference
Score 3: Both responses are roughly equivalent in quality
Score 4: Generated response is somewhat better than reference
Score 5: Generated response is significantly better than reference
"""


@run_spec_function("arena_hard_v01")
def get_arena_hard_v01_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.arena_hard_v01_scenario.ArenaHardV01Scenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "win_rate"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-turbo",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 1024,
                "metric_name": "win_rate",
                "rubric": _RUBRIC_WIN_RATE,
            },
        ),
    ]

    return RunSpec(
        name="arena_hard_v01",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "arena_hard_v01"],
        annotators=annotators,
    )
