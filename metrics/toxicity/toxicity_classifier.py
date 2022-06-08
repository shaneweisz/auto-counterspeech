from typing import List
from detoxify import Detoxify
import torch
from tqdm import tqdm
from ..metric import Metric
import numpy as np


class UnbiasedRoBERTaToxicityClassifier(Metric):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "Toxicity"

    def compute_score(self, predictions: List[str], verbose=False, **kwargs) -> float:
        model_type = "unbiased"  # "original" or "unbiased" or "multilingual"
        classifier = Detoxify(model_type, device=self.device)

        toxicity_scores = []
        for prediction in tqdm(predictions, disable=not verbose):
            results = classifier.predict(prediction)
            toxicity_scores.append(results["toxicity"])

        mean_toxicity = np.mean(toxicity_scores)

        self.score = mean_toxicity
        self.individual_scores = toxicity_scores

        return mean_toxicity


def interact():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_text = input("Enter text: ")
    while input_text != "":
        results = Detoxify("unbiased", device=device).predict(input_text)
        print(results)
        input_text = input("Enter text: ")


if __name__ == "__main__":
    interact()
