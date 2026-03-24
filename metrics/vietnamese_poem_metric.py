from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# TODO: fetch full Vietnamese phonetic analysis from
# https://github.com/Anshler/poem_generator/blob/master/utils/check_rule.py#L47
# and replace the stubs below with the complete implementation.


def _length_score(completion: str, reference_text: str) -> float:
    """Ratio of actual line count to expected line count from reference."""
    actual_lines = [l for l in completion.strip().splitlines() if l.strip()]
    expected_lines = [l for l in reference_text.strip().splitlines() if l.strip()]
    if not expected_lines:
        return 0.0
    return min(len(actual_lines) / len(expected_lines), 1.0)


def _tone_score(completion: str) -> float:
    """Fraction of syllables matching expected tonal pattern. Stub implementation."""
    # TODO: implement full Vietnamese tone checking using phonetic rules.
    return 0.0


def _rhyme_score(completion: str) -> float:
    """Fraction of line-ending pairs that rhyme (stub: exact last-word match)."""
    lines = [l.strip() for l in completion.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return 0.0
    # Check adjacent pairs for matching last word
    matches = 0
    pairs = 0
    for i in range(len(lines) - 1):
        last_a = lines[i].split()[-1].lower() if lines[i].split() else ""
        last_b = lines[i + 1].split()[-1].lower() if lines[i + 1].split() else ""
        if last_a and last_b and last_a == last_b:
            matches += 1
        pairs += 1
    return matches / pairs if pairs > 0 else 0.0


def _poem_score(length: float, tone: float, rhyme: float) -> float:
    """Weighted average of length, tone, and rhyme sub-scores."""
    return (length * 0.4 + tone * 0.3 + rhyme * 0.3)


class VietnamesePoemMetric(Metric):
    """Evaluates Vietnamese poem quality across length, tone, rhyme, and overall sub-scores."""

    VALID_METRICS = {"poem_score", "length_score", "tone_score", "rhyme_score"}

    def __init__(self, metric: str):
        super().__init__()
        assert metric in self.VALID_METRICS, f"metric must be one of {self.VALID_METRICS}"
        self.metric = metric

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        references = request_state.instance.references
        reference_text = references[0].output.text.strip() if references else ""

        length = _length_score(completion, reference_text)
        tone = _tone_score(completion)
        rhyme = _rhyme_score(completion)

        if self.metric == "length_score":
            score = length
        elif self.metric == "tone_score":
            score = tone
        elif self.metric == "rhyme_score":
            score = rhyme
        else:  # poem_score
            score = _poem_score(length, tone, rhyme)

        return [Stat(MetricName(self.metric)).add(score)]
