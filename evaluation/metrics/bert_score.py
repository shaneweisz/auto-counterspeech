from typing import List
from .base_metric import Metric
import bert_score
import numpy as np
import logging
import transformers


class BERTScore(Metric):
    def __init__(self) -> None:
        super().__init__()

        # hide the warning about some weights of the model checkpoint not used
        transformers.modeling_utils.logger.setLevel(logging.ERROR)
        self.bert_scorer = bert_score.BERTScorer(
            idf=True, lang="en", rescale_with_baseline=True
        )
        transformers.modeling_utils.logger.setLevel(logging.WARNING)

    @property
    def name(self) -> str:
        return f"BERTScore ({self.bert_scorer.hash})"

    def compute_score(self, predictions: List[str], **kwargs) -> float:
        assert "references" in kwargs
        references: List[str] = kwargs["references"]
        verbose: bool = kwargs.get("verbose", False)

        self.bert_scorer.compute_idf(sents=references)

        (P, R, F) = self.bert_scorer.score(predictions, references, verbose=verbose)

        mean_F = np.mean(F.tolist())
        return mean_F
