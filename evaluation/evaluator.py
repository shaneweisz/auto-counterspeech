from pathlib import Path
from typing import Dict, List
import pandas as pd
from .metrics import Metric


class Evaluator:
    def __init__(
        self,
        predictions: List[str],
        references: List[str] = None,
        inputs: List[str] = None,
    ):
        self.predictions = predictions
        if references:
            assert len(predictions) == len(references)
            self.references = references
        if inputs:
            assert len(predictions) == len(inputs)
            self.inputs = inputs

    def evaluate(self, metrics: List[Metric], verbose=False) -> Dict[str, float]:
        metric_scores = {}
        for metric in metrics:
            if verbose:
                print(f"Evaluating {metric.name}")
            score = metric.compute_score(
                predictions=self.predictions,
                references=self.references,
                verbose=verbose,
            )
            metric_scores[metric.name] = score
        return metric_scores

    @classmethod
    def from_files(
        cls,
        predictions_path: Path,
        references_path: Path = None,
        inputs_path: Path = None,
    ):
        predictions = [line.strip() for line in predictions_path.open()]
        if references_path:
            references = [line.strip() for line in references_path.open()]
        else:
            references = None
        if inputs_path:
            inputs = [line.strip() for line in inputs_path.open()]
        else:
            inputs = None
        return cls(predictions, references, inputs)

    @classmethod
    def from_csv(cls, csv_path: Path):
        df = pd.read_csv(csv_path)

        predictions = df["prediction"].to_list()
        references = df["reference"].to_list()
        inputs = df["input"].to_list()

        return cls(predictions, references, inputs)
