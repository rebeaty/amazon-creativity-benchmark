"""HELM Run Specs for webnovelbench."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_LITERARY_DEVICES = """\
Evaluate the use of LITERARY DEVICES in the generated web novel passage.
Consider use of metaphor, simile, imagery, foreshadowing, and other literary techniques.

Score 1: No literary devices used; completely plain and unadorned prose
Score 2: Minimal use of literary devices with poor execution
Score 3: Some literary devices used with moderate effectiveness
Score 4: Good use of literary devices that enhance the narrative
Score 5: Excellent, sophisticated use of literary devices that greatly enrich the prose
"""

_RUBRIC_LLM_JUDGE_CHARACTER_CONSISTENCY = """\
Evaluate the CHARACTER CONSISTENCY in the generated web novel passage.
Consider whether character voices, motivations, and behaviors are consistent throughout the passage.

Score 1: Characters behave inconsistently or contradict established traits
Score 2: Multiple character consistency issues throughout the passage
Score 3: Characters are generally consistent with some minor issues
Score 4: Characters are well-maintained throughout with good consistency
Score 5: Perfect character consistency with authentic and stable characterization
"""


@run_spec_function("webnovelbench")
def get_webnovelbench_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.webnovelbench_scenario.WebNovelBenchScenario",
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
        MetricSpec(class_name="metrics.percentile_rank_metric.PercentileRankMetric", args={}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_literary_devices"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_character_consistency"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "deepseek/deepseek-v3",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 1024,
                "metric_name": "llm_judge_literary_devices",
                "rubric": _RUBRIC_LLM_JUDGE_LITERARY_DEVICES,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "deepseek/deepseek-v3",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 1024,
                "metric_name": "llm_judge_character_consistency",
                "rubric": _RUBRIC_LLM_JUDGE_CHARACTER_CONSISTENCY,
            },
        ),
    ]

    return RunSpec(
        name="webnovelbench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "webnovelbench"],
        annotators=annotators,
    )
