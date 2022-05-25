from typing import List
from .base_metric import Metric
import datasets


class ROUGE(Metric):
    def __init__(self, rouge_type: str = "rouge2"):
        if rouge_type not in ["rouge1", "rouge2", "rougeL"]:
            raise ValueError(f"Unsupported rouge_type: {rouge_type}")
        self.rouge_type = rouge_type
        self.rouge = datasets.load_metric("rouge")

    @property
    def name(self) -> str:
        return f"ROUGE-{self.rouge_type[5:]}"

    def compute_score(self, predictions: List[str], references: List[str]) -> float:
        results = self.rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=[self.rouge_type],
        )
        if self.rouge_type == "rougeL":
            score = results[self.rouge_type].mid.fmeasure
        else:
            score = results[self.rouge_type].mid.recall
        return score
