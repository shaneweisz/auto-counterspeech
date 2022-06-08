from typing import List
from ..metric import Metric
import numpy as np


class ResponseLengthSummaryStatistic(Metric):
    def __init__(self, kind: str):
        super().__init__()
        assert kind in ["Max", "Min", "Avg", "Median"]
        self.kind = kind

    @property
    def name(self) -> str:
        return f"{self.kind.capitalize()}Len"

    def compute_score(self, predictions: List[str], **kwargs) -> float:
        summary_stat_func = None
        if self.kind == "Max":
            summary_stat_func = np.max
        elif self.kind == "Min":
            summary_stat_func = np.min
        elif self.kind == "Avg":
            summary_stat_func = np.mean
        elif self.kind == "Median":
            summary_stat_func = np.median

        lengths = [len(pred.split(" ")) for pred in predictions]
        summary_stat = summary_stat_func(lengths)
        self.score = summary_stat

        return summary_stat
