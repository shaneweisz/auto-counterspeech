from pathlib import Path
from typing import Dict, List
import pandas as pd
from .metrics import Metric


class Evaluator:
    def __init__(
        self,
        predictions: List[str],
        references: List[List[str]] = None,  # allows multiple references per prediction
        inputs: List[str] = None,
    ):
        self.predictions = predictions
        if references:
            assert len(predictions) == len(references)
            self.references = references
        if inputs:
            assert len(predictions) == len(inputs)
            self.inputs = inputs

    @classmethod
    def from_csv(cls, csv_path: Path):
        df = pd.read_csv(csv_path)

        predictions = df["prediction"].to_list()
        references = df["reference"].to_list()
        references = [[reference] for reference in references]
        inputs = df["input"].to_list()

        return cls(predictions, references, inputs)

    def evaluate(self, metrics: List[Metric]) -> Dict[str, float]:
        metric_scores = {}
        for metric in metrics:
            score = metric.compute_score(
                predictions=self.predictions, references=self.references
            )
            metric_scores[metric.name] = score
        return metric_scores
