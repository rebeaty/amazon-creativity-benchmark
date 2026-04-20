"""HELM Run Specs for dat_creative_writing."""

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
Evaluate the CREATIVITY of the creative writing piece generated for a Divergent Association Task (DAT) prompt.
Consider how creatively and divergently the writing connects the given words or concepts.

Score 1: Writing shows no creative or divergent thinking in connecting concepts
Score 2: Minimal creative connection; mostly obvious associations
Score 3: Some creative connections but mostly conventional
Score 4: Creative writing with notably divergent and interesting conceptual connections
Score 5: Highly creative writing demonstrating exceptional divergent thinking and novel conceptual connections
"""


@run_spec_function("dat_creative_writing")
def get_dat_creative_writing_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.dat_creative_writing_scenario.DATCreativeWritingScenario",
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
        MetricSpec(
            class_name="metrics.semantic_diversity_metric.SemanticDiversityMetric",
            args={"model_name": "all-mpnet-base-v2"},
        ),
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
        name="dat_creative_writing",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "dat_creative_writing"],
        annotators=annotators,
    )
