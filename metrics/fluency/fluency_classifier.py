from typing import List
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..metric import Metric
from nltk.tokenize import sent_tokenize


class RoBERTaColaFluencyClassifier(Metric):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self) -> None:
        super().__init__()
        model_name = "cointegrated/roberta-large-cola-krishna2020"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def name(self) -> str:
        return "Fluency"

    def compute_score(self, predictions: List[str], verbose: bool = False, **kwargs) -> float:
        scores = []
        for prediction in tqdm(predictions, disable=not verbose):
            sentence_scores = []
            for sentence in sent_tokenize(prediction):
                sentence_scores.append(self.score_sentence(sentence))
            score = np.mean(sentence_scores)
            scores.append(score)
        self.individual_scores = scores
        self.score = np.mean(scores)
        return self.score

    def score_sentence(self, sentence: str) -> float:
        encoded_sentence = self.tokenizer.encode_plus(sentence, return_tensors="pt")
        encoded_sentence = encoded_sentence.to(self.device)
        logits = self.model(**encoded_sentence).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        probability_fluent = probabilities[0][0].item()
        return probability_fluent
