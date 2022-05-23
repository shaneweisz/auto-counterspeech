from pathlib import Path
from typing import List
from datasets import load_metric
import pandas as pd


class Evaluator:
    def __init__(
        self,
        predictions: List[str],
        references: List[List[str]],
        inputs: List[str] = None,
    ):
        assert len(predictions) == len(references) == len(inputs)
        self.predictions = predictions
        self.references = references
        self.inputs = inputs

    @classmethod
    def from_csv(cls, csv_path: Path):
        df = pd.read_csv(csv_path)

        predictions = df["prediction"].to_list()
        references = df["reference"].to_list()
        references = [[r] for r in references]
        inputs = df["input"].to_list()

        return cls(predictions, references, inputs)

    @classmethod
    def from_json(cls, json_path: Path):
        # TODO: allow for multiple references through a JSON file
        pass

    def evaluate(self, metrics: List[str]) -> float:
        for metric in metrics:
            metric_obj = load_metric(metric)
            metric_value = metric_obj.compute(
                predictions=self.predictions, references=self.references
            )
            print(f"{metric}: {metric_value}")
