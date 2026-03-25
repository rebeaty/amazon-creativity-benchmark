"""HELM Run Specs for pron_vs_prompt."""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import (
    ADAPT_GENERATION,
)
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── Rubrics ──────────────────────────────────────────────────────────────────

_RUBRIC_LLM_JUDGE_ATTRACTIVENESS = """\
Evaluate the ATTRACTIVENESS of the generated promotional or advertising text.
Consider how appealing, engaging, and attention-grabbing the text would be to its target audience.

Score 1: Text is unappealing and would fail to attract any audience attention
Score 2: Minimally attractive with poor engagement qualities
Score 3: Moderately attractive with some appealing elements
Score 4: Genuinely attractive text that would engage the target audience
Score 5: Highly attractive, attention-grabbing text with excellent audience appeal
"""

_RUBRIC_LLM_JUDGE_ORIGINALITY = """\
Evaluate the ORIGINALITY of the generated promotional or advertising text.
Consider how novel, fresh, and creative the text is compared to typical advertising copy.

Score 1: Text is a cliche or generic advertising copy with no originality
Score 2: Minimal originality; uses common advertising tropes
Score 3: Some original elements within mostly conventional advertising language
Score 4: Notably original text that stands out from typical advertising
Score 5: Highly original, innovative promotional text with exceptional creative freshness
"""

_RUBRIC_LLM_JUDGE_CREATIVITY = """\
Evaluate the CREATIVITY of the generated promotional or advertising text.
Consider creative use of language, metaphor, wordplay, and imaginative approaches to the advertising message.

Score 1: No creative elements; completely mechanical advertising copy
Score 2: Minimal creativity with mostly standard advertising language
Score 3: Some creative elements with moderate linguistic imagination
Score 4: Genuinely creative text with strong original linguistic elements
Score 5: Highly creative text demonstrating exceptional advertising creativity and originality
"""


@run_spec_function("pron_vs_prompt")
def get_pron_vs_prompt_spec() -> RunSpec:

    scenario_spec = ScenarioSpec(
        class_name="scenarios_new.pron_vs_prompt_scenario.PronVsPromptScenario",
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
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_attractiveness"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_originality"}),
        MetricSpec(class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric", args={"metric_name": "llm_judge_creativity"}),
    ]

    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_attractiveness",
                "rubric": _RUBRIC_LLM_JUDGE_ATTRACTIVENESS,
            },
        ),
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": "openai/gpt-4",
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 256,
                "metric_name": "llm_judge_originality",
                "rubric": _RUBRIC_LLM_JUDGE_ORIGINALITY,
            },
        ),
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
        name="pron_vs_prompt",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "pron_vs_prompt"],
        annotators=annotators,
    )
