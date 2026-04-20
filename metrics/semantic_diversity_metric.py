"""
SemanticDiversityMetric: per-instance semantic diversity via DSI.

For DAT-style responses (10 unrelated nouns), computes the mean pairwise
cosine distance between word embeddings within a single response.
This is the within-response analog of semantic diversity: higher = more creative.

Reuses segmentation and DSI helpers from creativity_score_metric.py.
"""

import threading
from typing import List, Optional

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

from metrics.creativity_score_metric import _segment_response, _compute_dsi


class SemanticDiversityMetric(Metric):
    """Mean pairwise cosine distance between segment embeddings of a single response.

    Args:
        model_name: SentenceTransformer model identifier.
        task: Segmentation scheme passed to _segment_response.
              Use "dat" for DAT (word lists), "cwt" for sentences, etc.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        task: str = "dat",
    ):
        super().__init__()
        self.model_name = model_name
        self.task = task
        self._model: Optional[object] = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(
                        self.model_name,
                        device="cpu",
                        model_kwargs={"low_cpu_mem_usage": False},
                    )
        return self._model

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        completion = request_state.result.completions[0].text.strip()

        if not completion:
            return [Stat(MetricName("semantic_diversity")).add(0.0)]

        segments = _segment_response(completion, self.task)
        model = self._get_model()
        score = _compute_dsi(segments, model)

        return [Stat(MetricName("semantic_diversity")).add(score)]

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat


class SemanticDiversityMetric(Metric):
    """Measures semantic diversity within a single output using SentenceTransformers.

    For word-list outputs (DAT tasks): embeds each word and computes mean pairwise cosine distance.
    For prose outputs (writing tasks): embeds each sentence and computes mean pairwise cosine distance.
    Score range: [0, 1], higher means more semantically diverse.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2", mode: str = "auto"):
        super().__init__()
        self._model_name = model_name
        self._mode = mode  # "words", "sentences", or "auto"
        self._model = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name,
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False},
            )
        return self._model

    @staticmethod
    def _mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
        n = len(embeddings)
        if n < 2:
            return 0.0
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = embeddings / norms
        sim_matrix = normed @ normed.T
        # Extract upper triangle (excluding diagonal)
        upper = sim_matrix[np.triu_indices(n, k=1)]
        mean_sim = float(np.mean(upper))
        return 1.0 - mean_sim  # distance = 1 - similarity

    def _parse_units(self, text: str) -> List[str]:
        if self._mode == "words":
            units = [t.strip(".,;:-\n\"'") for t in text.split()]
            return [u for u in units if u]
        if self._mode == "sentences":
            units = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            return units or [text]
        # auto: use words if output looks like a word list (short tokens), else sentences
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Heuristic: if most lines are single tokens, treat as word list
        single_token_lines = sum(1 for ln in lines if " " not in ln)
        if lines and single_token_lines / len(lines) > 0.5:
            return [ln.strip(".,;:-\n\"'") for ln in lines if ln.strip()]
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        return sentences or [text]

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        assert request_state.result is not None
        prediction = request_state.result.completions[0].text.strip()

        units = self._parse_units(prediction)
        if len(units) < 2:
            return [Stat(MetricName("semantic_diversity")).add(0.0)]

        model = self._get_model()
        embeddings = model.encode(units, convert_to_numpy=True)
        score = self._mean_pairwise_cosine_distance(embeddings)

        return [Stat(MetricName("semantic_diversity")).add(score)]
