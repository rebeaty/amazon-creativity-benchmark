"""HELM Run Specs for dialogue_diversity."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_COHERENCE_SCORE = """\
Evaluate the COHERENCE of the generated dialogue response.
Consider whether the response logically follows from the conversation context and maintains conversational coherence.

Score 1: Response is completely incoherent or off-topic for the dialogue
Score 2: Response has major coherence issues with the preceding conversation
Score 3: Response is somewhat coherent but with noticeable gaps
Score 4: Response is coherent and follows naturally from the dialogue
Score 5: Response is perfectly coherent, natural, and contextually appropriate
"""


@run_spec_function("dialogue_diversity")
def get_dialogue_diversity_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.dialogue_diversity_scenario.DialogueDiversityScenario",
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
        MetricSpec(class_name="metrics.distinct_n_metric.DistinctNMetric", args={"n": 1}),
        MetricSpec(class_name="metrics.distinct_n_metric.DistinctNMetric", args={"n": 2}),
        MetricSpec(
            class_name="metrics.semantic_diversity_metric.SemanticDiversityMetric",
            args={"model_name": "all-mpnet-base-v2", "task": "cwt"},
        ),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "coherence_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 64,
                "metric_name": "coherence_score",
                "rubric": _RUBRIC_COHERENCE_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="dialogue_diversity",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "dialogue_diversity"],
        annotators=annotators,
    )
