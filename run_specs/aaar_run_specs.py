"""HELM Run Specs for aaar."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_RECALL_GT_ENTAIL_SCORE = """\
Evaluate whether the generated research content covers the key points from the ground-truth reference.
For each key point in the reference, assess whether it is entailed (covered) by the generated text.

Score 1: Generated text covers almost none of the reference key points
Score 2: Generated text covers less than half the reference key points
Score 3: Generated text covers roughly half the reference key points
Score 4: Generated text covers most of the reference key points
Score 5: Generated text fully covers all key points from the reference
"""

_RUBRIC_PRECISION_PRED_ENTAIL_SCORE = """\
Evaluate the precision of the generated research content relative to the ground-truth reference.
Assess what fraction of the generated points are actually entailed by (relevant to) the reference.

Score 1: Almost all generated points are irrelevant to the reference
Score 2: Most generated points are irrelevant or incorrect
Score 3: About half the generated points are relevant and correct
Score 4: Most generated points are relevant and supported by the reference
Score 5: All generated points are precisely entailed by the reference
"""


@run_spec_function("aaar_experiment_design")
def get_aaar_experiment_design_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.aaar_scenario.AaarScenario",
        args={"subtask": "experiment_design"},
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
        MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric", args={"model_name": "all-mpnet-base-v2"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "recall_gt_entail_score"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "precision_pred_entail_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-1106-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "recall_gt_entail_score",
                "rubric": _RUBRIC_RECALL_GT_ENTAIL_SCORE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-1106-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "precision_pred_entail_score",
                "rubric": _RUBRIC_PRECISION_PRED_ENTAIL_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="aaar:subtask=experiment_design",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "aaar"],
        annotators=annotators,
    )


@run_spec_function("aaar_paper_weakness")
def get_aaar_paper_weakness_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.aaar_scenario.AaarScenario",
        args={"subtask": "paper_weakness"},
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
        MetricSpec(class_name="helm.benchmark.metrics.summarization_metrics.SummarizationMetric", args={"model_name": "all-mpnet-base-v2"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "recall_gt_entail_score"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "precision_pred_entail_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-1106-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "recall_gt_entail_score",
                "rubric": _RUBRIC_RECALL_GT_ENTAIL_SCORE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4-1106-preview",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "precision_pred_entail_score",
                "rubric": _RUBRIC_PRECISION_PRED_ENTAIL_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="aaar:subtask=paper_weakness",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "aaar"],
        annotators=annotators,
    )
