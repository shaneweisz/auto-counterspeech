from __future__ import annotations
from typing import List


class Metric:
    def __init__(self):
        self.score = None
        self.individual_scores = None

    @property
    def name(self) -> str:
        pass

    def compute_score(
        self,
        predictions: List[str],
        **metric_kwargs,
    ) -> float:
        """Compute the score of the metric, and store it in `self.score`.
        Optionally also store a score for each prediction in `self.detailed_scores`."""
        pass
