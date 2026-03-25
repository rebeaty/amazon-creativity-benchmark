"""HELM Run Specs for historical_analogy."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_JUDGE_SCORE_ANALOGY = """\
Evaluate the quality of the historical analogy generated.
Consider the accuracy of historical facts, appropriateness of the analogy, insightfulness, and persuasiveness.

Score 1: Analogy is historically inaccurate or completely inappropriate
Score 2: Weak analogy with significant historical errors or poor fit
Score 3: Adequate analogy that is mostly accurate but lacks depth
Score 4: Good analogy with accurate historical details and clear relevance
Score 5: Excellent analogy that is historically accurate, insightful, and highly persuasive
"""


@run_spec_function("historical_analogy")
def get_historical_analogy_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.historical_analogy_scenario.HistoricalAnalogyScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "judge_score_analogy"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "judge_score_analogy",
                "rubric": _RUBRIC_JUDGE_SCORE_ANALOGY,
            },
        ),
    ]

    return RunSpec(
        name="historical_analogy",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "historical_analogy"],
        annotators=annotators,
    )
