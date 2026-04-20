"""F1 metric: token-level F1 between prediction and gold reference, stat named 'f1'."""

from typing import List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.evaluate_reference_metrics import f1_score
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class F1Metric(Metric):
    """Token-level F1 score between the first prediction and gold references.

    Produces a stat named 'f1' (not 'f1_score') to match registry expectations.
    Score is the max F1 across all gold references.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        references = request_state.instance.references
        if not references:
            return [Stat(MetricName("f1")).add(0.0)]

        pred = request_state.result.completions[0].text.strip()
        score = max(f1_score(ref.output.text, pred) for ref in references)
        return [Stat(MetricName("f1")).add(score)]

import re
import string
from typing import List

from nltk.metrics.scores import f_measure

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.scenarios.scenario import CORRECT_TAG


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


class F1Metric(Metric):
    """Token-level F1 between prediction and reference, emitted as stat named 'f1'."""

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        pred = request_state.result.completions[0].text.strip()

        correct_refs = [
            ref.output.text
            for ref in request_state.instance.references
            if CORRECT_TAG in ref.tags
        ]
        if not correct_refs:
            return [Stat(MetricName("f1")).add(0.0)]

        best = max(
            f_measure(set(_normalize(gold).split()), set(_normalize(pred).split())) or 0.0
            for gold in correct_refs
        )
        return [Stat(MetricName("f1")).add(best)]
