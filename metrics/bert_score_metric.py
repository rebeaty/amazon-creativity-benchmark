"""BERTScore metric: F1 BERTScore between prediction and gold references."""

import threading
from typing import List, Optional

from bert_score import BERTScorer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

_scorer_lock = threading.Lock()


class BertScoreMetric(Metric):
    """BERTScore F1 between the first prediction and gold references.

    Produces a stat named 'bert_score'. Uses bert-base-uncased by default.
    Score is the max BERTScore-F1 across all gold references.
    """

    def __init__(self, model_type: str = "bert-base-uncased", device: str = "cpu"):
        self.model_type = model_type
        self.device = device
        self._scorer: Optional[BERTScorer] = None

    def _load_scorer(self) -> None:
        with _scorer_lock:
            if self._scorer is not None:
                return
            import transformers
            # Newer transformers defaults low_cpu_mem_usage=True, which loads weights
            # onto meta device. bert_score then calls model.to(device) which fails on
            # meta tensors. Force eager loading to avoid this.
            _orig = transformers.AutoModel.from_pretrained

            def _eager_from_pretrained(*args, **kwargs):
                kwargs["low_cpu_mem_usage"] = False
                return _orig(*args, **kwargs)

            transformers.AutoModel.from_pretrained = _eager_from_pretrained
            try:
                self._scorer = BERTScorer(model_type=self.model_type, device=self.device)
            finally:
                transformers.AutoModel.from_pretrained = _orig

    def _compute_f1(self, pred: str, ref: str) -> float:
        self._load_scorer()
        assert self._scorer is not None
        _, _, F = self._scorer.score([pred], [ref])
        return F[0].item()

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
            return [Stat(MetricName("bert_score")).add(0.0)]

        pred = request_state.result.completions[0].text.strip()
        refs = [ref.output.text for ref in references]

        best_f1 = max(self._compute_f1(pred, ref) for ref in refs)
        return [Stat(MetricName("bert_score")).add(best_f1)]
