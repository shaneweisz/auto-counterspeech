from typing import List
from .base_metric import Metric
import nltk
from nltk.util import ngrams


class DistinctN(Metric):
    def __init__(self, N=1) -> None:
        super().__init__()
        self.N = N

    @property
    def name(self) -> str:
        return f"Distinct-{self.N}"

    def compute_score(self, predictions: List[str], **kwargs) -> float:
        distinct_ngrams = set()
        total_number_of_ngrams = 0

        for prediction in predictions:
            for ngram in ngrams(nltk.word_tokenize(prediction), self.N):
                total_number_of_ngrams += 1
                distinct_ngrams.add(ngram)

        score = len(distinct_ngrams) / total_number_of_ngrams
        return score
