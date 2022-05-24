from typing import List
from nltk.translate.bleu_score import corpus_bleu
from .base_metric import Metric
import nltk


class BLEU(Metric):
    def __init__(self, N=4):
        self.N = N

    @property
    def name(self) -> str:
        return f"BLEU-{self.N}"

    def compute_score(
        self, predictions: List[str], references: List[List[str]]
    ) -> float:
        predictions_tok = [nltk.word_tokenize(pred) for pred in predictions]
        references_tok = [[nltk.word_tokenize(r) for r in rs] for rs in references]
        weights = tuple([1 / self.N] * self.N)  # e.g. (0.5, 0.5) for N=2
        score = corpus_bleu(references_tok, predictions_tok, weights)
        return score
