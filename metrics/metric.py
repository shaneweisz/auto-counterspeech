from __future__ import annotations
from typing import List


class Metric:
    def __init__(self):
        self._score = None
        self.individual_scores = None

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def score(self):
        if self._score is None:
            raise ValueError("Score not computed yet. Call `compute_score()` first.")
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    def compute_score(
        self,
        predictions: List[str],
        verbose: bool = False,
        **metric_kwargs,
    ) -> float:
        """Compute the score of the metric, and store it in `self.score`.
        Optionally also store an individual score for each prediction in `self.individual_scores`."""
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.name+':':<16} {self.score:.3f}"
