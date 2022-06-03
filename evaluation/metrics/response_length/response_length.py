from typing import List
from ..base_metric import Metric
import numpy as np
import nltk


class ResponseLengthSummaryStatistic(Metric):
    def __init__(self, kind: str):
        assert kind in ["max", "min", "mean", "median"]
        self.kind = kind

    @property
    def name(self) -> str:
        return f"{self.kind} response length"

    def compute_score(self, predictions: List[str], **kwargs) -> float:
        summary_stat_func = None
        if self.kind == "max":
            summary_stat_func = np.max
        elif self.kind == "min":
            summary_stat_func = np.min
        elif self.kind == "mean":
            summary_stat_func = np.mean
        elif self.kind == "median":
            summary_stat_func = np.median

        return summary_stat_func([len(nltk.word_tokenize(pred)) for pred in predictions])
