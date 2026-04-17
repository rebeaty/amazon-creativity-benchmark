"""HELM Run Specs for macgyver."""

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
Evaluate the CORRECTNESS of the generated MacGyver-style creative problem solution.
Consider whether the solution is physically plausible, uses the specified materials, and would actually work.

Score 1: Solution is physically impossible or uses unavailable materials
Score 2: Solution has major feasibility issues or incorrect material use
Score 3: Solution is plausible in principle but with significant practical gaps
Score 4: Solution is mostly correct and feasible with the given materials
Score 5: Solution is fully correct, physically plausible, and creative within constraints
"""


@run_spec_function("macgyver")
def get_macgyver_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.macgyver_scenario.MacgyverScenario",
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
        MetricSpec(class_name="helm.benchmark.metrics.basic_metrics.BasicGenerationMetric", args={"names": ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"]}),
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
        name="macgyver",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "macgyver"],
        annotators=annotators,
    )
