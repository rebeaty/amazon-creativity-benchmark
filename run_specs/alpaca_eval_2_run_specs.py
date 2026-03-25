"""HELM Run Specs for alpaca_eval_2."""

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
Compare the generated response against a reference response for an instruction-following task.
Determine which response is better overall in terms of helpfulness, quality, and instruction-following.

Score 1: Generated response is significantly worse than reference
Score 2: Generated response is somewhat worse than reference
Score 3: Generated response is roughly equal to reference
Score 4: Generated response is somewhat better than reference
Score 5: Generated response is significantly better than reference
"""


@run_spec_function("alpaca_eval_2")
def get_alpaca_eval_2_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.alpaca_eval_2_scenario.AlpacaEval2Scenario",
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
        name="alpaca_eval_2",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "alpaca_eval_2"],
        annotators=annotators,
    )
