from typing import List
from ..base_metric import Metric
import bert_score
import numpy as np
import logging
import transformers


class BERTScore(Metric):
    @property
    def name(self) -> str:
        return "BERTScore"

    def compute_score(self, predictions: List[str], references: List[str], verbose: bool, **kwargs) -> float:
        bert_scorer = self._load_bert_scorer(references)
        (P, R, F) = bert_scorer.score(predictions, references, verbose=verbose)

        self.individual_scores = F.tolist()
        self.score = np.mean(self.individual_scores)

        return self.score

    def _load_bert_scorer(self, references: List[str]):
        too_few_refs_for_idf = len(references) <= 100
        idf = False if too_few_refs_for_idf else True
        idf_sents = references if idf else None

        # hide the warning about some weights of the model checkpoint not used
        transformers.modeling_utils.logger.setLevel(logging.ERROR)
        bert_scorer = bert_score.BERTScorer(
            idf=idf,
            idf_sents=idf_sents,
            lang="en",
            rescale_with_baseline=True,
        )
        transformers.modeling_utils.logger.setLevel(logging.WARNING)

        return bert_scorer
