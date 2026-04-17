"""HELM Run Specs for tinystories."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_GRAMMAR_SCORE = """\
Evaluate the GRAMMAR quality of the generated children's short story.
Consider grammatical correctness, sentence structure, and linguistic clarity appropriate for children.

Score 1: Severe grammatical errors throughout that make the text unreadable
Score 2: Frequent grammatical errors impeding comprehension
Score 3: Some grammatical issues but generally understandable
Score 4: Minor grammatical issues with mostly correct and clear language
Score 5: Perfect grammar appropriate for children's literature
"""

_RUBRIC_CREATIVITY_SCORE = """\
Evaluate the CREATIVITY of the generated children's short story.
Consider originality of plot, imaginative characters, and creative storytelling appropriate for children.

Score 1: Completely generic and predictable story with no creative elements
Score 2: Minimal creativity; follows common children's story clichés throughout
Score 3: Some creative elements mixed with conventional storytelling
Score 4: Genuinely creative story with original plot or character elements
Score 5: Highly creative, imaginative story that would delight children with its originality
"""

_RUBRIC_CONSISTENCY_SCORE = """\
Evaluate the CONSISTENCY of the generated children's short story.
Consider character consistency, plot logic, and internal coherence throughout the story.

Score 1: Story has major inconsistencies in characters, plot, or setting
Score 2: Multiple inconsistencies that disrupt the reading experience
Score 3: Generally consistent with some minor continuity issues
Score 4: Consistent throughout with at most one minor inconsistency
Score 5: Perfectly consistent story with coherent characters, plot, and setting
"""


@run_spec_function("tinystories")
def get_tinystories_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios.tinystories_scenario.TinyStoriesScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "grammar_score"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "creativity_score"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "consistency_score"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "grammar_score",
                "rubric": _RUBRIC_GRAMMAR_SCORE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "creativity_score",
                "rubric": _RUBRIC_CREATIVITY_SCORE,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "consistency_score",
                "rubric": _RUBRIC_CONSISTENCY_SCORE,
            },
        ),
    ]

    return RunSpec(
        name="tinystories",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "tinystories"],
        annotators=annotators,
    )
