"""HELM Run Specs for the CAP (Creativity Assessment Platform) battery.

Five tasks from the UVA pilot / Study 3 with identical Gemini "V7 quality"
judge rubric across all of them (same rubric the in-repo
[human/uva_pilot/scripts/old/score_quality_v7_pilot.py] uses on the
human data — keeps human/LLM scores on the same scale).

Quality judge: google/gemini-3-flash-preview, temperature=0, 1-7 scale,
rates WHETHER a response makes sense (not creativity/originality).
Novelty is NOT computed here — it's a cross-corpus metric that needs the
full pool. See [scripts/score_cap_novelty.py] for post-HELM aggregation.

Note: pool centroid for novelty will eventually use the 200-model ABC
corpus. For now the existing UVA pilot pool stands in as a proxy — see
the CAP README for the caveat.

Run spec functions (one per task):
  - cap_aut        — 5 objects, multi-response (3-6 ideas per prompt)
  - cap_sctt       — 5 prompts, multi-response
  - cap_design     — 5 prompts, multi-response
  - cap_metaphor   — 10 stems, single-response (short phrase)
  - cap_story      — 5 triads, single-response (3-8 sentence story)
"""

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.adapters.adapter_factory import ADAPT_GENERATION
from helm.benchmark.annotation.annotator import AnnotatorSpec
from helm.benchmark.metrics.metric import MetricSpec
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec


# ── V7 Quality Rubric (matches the human scoring pipeline) ──────────────────
# The score_quality_v7_pilot.py script judges each individual idea in
# multi-response tasks (splitting on ';') AND single-response wholes.
# Inside HELM the judge sees the whole completion (semicolon-separated for
# AUT/SCTT/Design), so we use the "single-response" V7 variant for those
# as well — the rubric still asks "does this response make sense". A
# post-hoc per-idea re-judge can be added later using the per-idea rubric
# and the same judge model if item-level quality granularity is needed.

_V7_QUALITY_RUBRIC = """\
Does this response make sense as a response to the prompt?

Rate on a 1-7 scale:
1 = Invalid (blank, gibberish, or completely unrelated to the prompt)
2 = Nonsensical (attempts to address the prompt but the response doesn't make sense)
3 = Questionable (the response is understandable but has clear logical or practical problems)
4 = Passable (the response makes sense but is vague or only loosely addresses the prompt)
5 = Sound (the response makes sense and addresses the prompt)
6 = Strong (the response clearly makes sense and directly addresses the prompt)
7 = Excellent (the response makes perfect sense as a response to this prompt)

IMPORTANT:
- Rate ONLY whether the response makes sense. Do NOT judge creativity, originality, or cleverness.
- A brief response and an elaborate response should get the SAME score if the core content is equally sound.
- Do NOT reward detail, literary quality, polish, or length.
"""


_JUDGE_MODEL = "google/gemini-3-flash-preview"
_JUDGE_METRIC_NAME = "cap_quality"


def _build_spec(
    scenario_class: str,
    run_spec_name: str,
    max_tokens: int,
    num_repetitions: int = 1,
) -> RunSpec:
    scenario_spec = ScenarioSpec(
        class_name=scenario_class,
        args={"num_repetitions": num_repetitions},
    )
    adapter_spec = AdapterSpec(
        method=ADAPT_GENERATION,
        instructions="",
        input_prefix="",
        input_suffix="\n",
        output_prefix="",
        output_suffix="\n",
        max_train_instances=0,
        num_outputs=1,
        max_tokens=max_tokens,
        temperature=0.7,
        stop_sequences=[],
    )
    metric_specs = [
        MetricSpec(
            class_name="llm_judge.generic_llm_judge_metric.GenericLLMJudgeMetric",
            args={"metric_name": _JUDGE_METRIC_NAME},
        ),
    ]
    annotators = [
        AnnotatorSpec(
            class_name="llm_judge.generic_llm_judge_annotator.GenericLLMJudgeAnnotator",
            args={
                "judge_model_name": _JUDGE_MODEL,
                "judge_temperature": 0.0,
                "judge_max_new_tokens": 80,
                "metric_name": _JUDGE_METRIC_NAME,
                "rubric": _V7_QUALITY_RUBRIC,
            },
        ),
    ]
    return RunSpec(
        name=run_spec_name,
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["creativity", "cap", run_spec_name],
        annotators=annotators,
    )


@run_spec_function("cap_aut")
def get_cap_aut_spec() -> RunSpec:
    return _build_spec(
        scenario_class="scenarios.cap_aut_scenario.CapAutScenario",
        run_spec_name="cap_aut",
        max_tokens=512,       # 3-6 ideas @ ~40 tokens each + separators
    )


@run_spec_function("cap_sctt")
def get_cap_sctt_spec() -> RunSpec:
    return _build_spec(
        scenario_class="scenarios.cap_sctt_scenario.CapSCTTScenario",
        run_spec_name="cap_sctt",
        max_tokens=512,
    )


@run_spec_function("cap_design")
def get_cap_design_spec() -> RunSpec:
    return _build_spec(
        scenario_class="scenarios.cap_design_scenario.CapDesignScenario",
        run_spec_name="cap_design",
        max_tokens=512,
    )


@run_spec_function("cap_metaphor")
def get_cap_metaphor_spec() -> RunSpec:
    return _build_spec(
        scenario_class="scenarios.cap_metaphor_scenario.CapMetaphorScenario",
        run_spec_name="cap_metaphor",
        max_tokens=80,        # 1-5 words per Study 3
    )


@run_spec_function("cap_story")
def get_cap_story_spec() -> RunSpec:
    return _build_spec(
        scenario_class="scenarios.cap_story_scenario.CapStoryScenario",
        run_spec_name="cap_story",
        max_tokens=512,       # 3-8 sentences per Study 3
    )
