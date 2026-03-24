import json
from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# TODO: Replace with full composite layout quality evaluation from:
# https://github.com/yizhiwang96/TextLogoLayout
# Full metric includes: non-overlapping ratio + alignment score + size-balance.


def _non_overlap_ratio(boxes: list) -> float:
    """Fraction of box pairs that do not overlap."""
    if len(boxes) < 2:
        return 1.0
    overlapping = 0
    total_pairs = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]):
                overlapping += 1
            total_pairs += 1
    return 1.0 - (overlapping / total_pairs)


class LayoutQualityMetric(Metric):
    """Composite layout quality: non-overlapping ratio as primary heuristic.

    Stub implementation. Replace with full TextLogoLayout evaluation when available.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        try:
            # Expect completion to be a JSON list of [x1, y1, x2, y2] bounding boxes
            boxes = json.loads(completion)
            if isinstance(boxes, list) and all(isinstance(b, list) and len(b) == 4 for b in boxes):
                score = _non_overlap_ratio(boxes)
            else:
                score = 0.0
        except (json.JSONDecodeError, TypeError):
            score = 0.0

        return [Stat(MetricName("layout_quality")).add(score)]
