from typing import List
from .base_metric import Metric
import datasets


class BERTScore(Metric):
    def __init__(self):
        self.bertscore = datasets.load_metric("bertscore")

    @property
    def name(self) -> str:
        return "BERTScore"

    def compute_score(
        self, predictions: List[str], references: List[List[str]]
    ) -> float:
        results = self.bertscore.compute(
            predictions=predictions, references=references, lang="en"
        )
        score_per_prediction = results["f1"]
        mean_score = sum(score_per_prediction) / len(score_per_prediction)
        return mean_score
