"""HELM Run Specs for showerthoughts."""

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
Evaluate the CREATIVITY of the generated showerthought.
Consider originality, unexpectedness, and how surprising or mind-bending the observation is.

Score 1: Thought is completely ordinary and unsurprising
Score 2: Slightly unusual but mostly conventional observation
Score 3: Somewhat creative thought with a mildly surprising element
Score 4: Genuinely creative thought that offers a fresh, surprising perspective
Score 5: Highly creative, mind-bending thought that offers an exceptionally original and surprising insight
"""

_RUBRIC_LLM_JUDGE_HUMOR = """\
Evaluate the HUMOR of the generated showerthought.
Consider how funny, witty, or amusing the thought is.

Score 1: Thought has no humor whatsoever
Score 2: Minimally amusing with weak comedic elements
Score 3: Mildly funny with some comedic value
Score 4: Genuinely funny thought that would amuse most readers
Score 5: Highly funny, witty thought with exceptional comedic quality
"""

_RUBRIC_LLM_JUDGE_CLEVERNESS = """\
Evaluate the CLEVERNESS of the generated showerthought.
Consider how intellectually sharp, insightful, and perceptive the observation is.

Score 1: Thought shows no cleverness or intellectual insight
Score 2: Minimally clever with little intellectual depth
Score 3: Moderately clever with some intellectual sharpness
Score 4: Genuinely clever thought demonstrating intellectual insight
Score 5: Exceptionally clever, insightful thought demonstrating outstanding intellectual sharpness
"""


@run_spec_function("showerthoughts")
def get_showerthoughts_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.showerthoughts_scenario.ShowerthoughtsScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_creativity"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_humor"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_cleverness"}),
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
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_humor",
                "rubric": _RUBRIC_LLM_JUDGE_HUMOR,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_cleverness",
                "rubric": _RUBRIC_LLM_JUDGE_CLEVERNESS,
            },
        ),
    ]

    return RunSpec(
        name="showerthoughts",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "showerthoughts"],
        annotators=annotators,
    )
