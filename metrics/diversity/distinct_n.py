from typing import List
from ..metric import Metric
import nltk
from nltk.util import ngrams


class DistinctN(Metric):
    def __init__(self, N=1) -> None:
        super().__init__()
        self.N = N

    @property
    def name(self) -> str:
        return f"Dist-{self.N}"

    def compute_score(self, predictions: List[str], **kwargs) -> float:
        distinct_ngrams = set()
        for prediction in predictions:
            for ngram in ngrams(nltk.word_tokenize(prediction), self.N):
                distinct_ngrams.add(ngram)

        num_distinct_ngrams = len(distinct_ngrams)
        total_number_of_words = sum([len(nltk.word_tokenize(prediction)) for prediction in predictions])

        score = num_distinct_ngrams / total_number_of_words
        self.score = score

        return score
