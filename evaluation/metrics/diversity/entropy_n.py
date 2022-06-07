from typing import List
from ..base_metric import Metric
import nltk
from nltk.util import ngrams
from math import log


class EntropyN(Metric):
    """
    Diversity metric that measures the evenness of the empirical n-gram frequency
    distribution.

    Flat distributions (high diversity) have higher entropy than highly peaked
    distributions (low diversity).
    """

    def __init__(self, N=2):
        super().__init__()
        self.N = N

    @property
    def name(self) -> str:
        return f"Ent-{self.N}"

    def compute_score(self, predictions: List[str], **kwargs) -> float:
        ngram_counts = self._count_ngrams(predictions)
        ngram_frequencies = self._ngram_frequencies(ngram_counts)
        score = self.entropy(ngram_frequencies.values())
        self.score = score
        return score

    def _count_ngrams(self, predictions: List[str]) -> dict:
        ngram_counts = {}

        for prediction in predictions:
            for ngram in ngrams(nltk.word_tokenize(prediction), self.N):
                if ngram not in ngram_counts:
                    ngram_counts[ngram] = 0
                ngram_counts[ngram] += 1

        return ngram_counts

    def _ngram_frequencies(self, ngram_counts: dict) -> dict:
        total_ngrams = sum(ngram_counts.values())
        ngram_frequencies = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
        return ngram_frequencies

    @staticmethod
    def entropy(probabilities: List[float]) -> float:
        return -sum([p * log(p) for p in probabilities])
