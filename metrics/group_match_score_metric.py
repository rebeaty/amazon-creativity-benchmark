from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class GroupMatchScoreMetric(Metric):
    """Jaccard similarity between predicted and reference token sets (group matching)."""

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

        # Parse as comma-separated items if present, else whitespace tokens
        def _parse(text: str):
            if "," in text:
                return set(item.strip().lower() for item in text.split(",") if item.strip())
            return set(text.lower().split())

        pred_set = _parse(completion)
        ref_set = _parse(reference_text)

        intersection = len(pred_set & ref_set)
        union = len(pred_set | ref_set)
        score = intersection / union if union > 0 else 0.0

        return [Stat(MetricName("group_match_score")).add(score)]
