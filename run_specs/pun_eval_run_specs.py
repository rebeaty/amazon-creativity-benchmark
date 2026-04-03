"""HELM Run Specs for pun_eval."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_QUALITY = """\
Evaluate the quality of the generated pun.
Consider cleverness of wordplay, humor effectiveness, grammatical correctness, and overall pun quality.

Score 1: Text is not a pun or fails completely as wordplay
Score 2: Poor pun with weak wordplay or poor execution
Score 3: Adequate pun with clear wordplay but limited humor impact
Score 4: Good pun with clever wordplay and effective humor
Score 5: Excellent pun demonstrating highly clever, original, and funny wordplay
"""


@run_spec_function("pun_eval")
def get_pun_eval_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.pun_eval_scenario.PunEvalScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_quality"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_quality",
                "rubric": _RUBRIC_LLM_JUDGE_QUALITY,
            },
        ),
    ]

    return RunSpec(
        name="pun_eval",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "pun_eval"],
        annotators=annotators,
    )
