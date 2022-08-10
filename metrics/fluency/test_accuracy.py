import pandas as pd
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Read in TSV
filename = sys.argv[1]
df = pd.read_csv(filename, sep="\t", header=None)
print("Number of rows:", df)
print(df.head())


# Load classifier (manually copied here to avoid path issues)
class RoBERTaColaFluencyClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self) -> None:
        super().__init__()
        model_name = "cointegrated/roberta-large-cola-krishna2020"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def score_sentence(self, sentence: str) -> float:
        encoded_sentence = self.tokenizer.encode_plus(sentence, return_tensors="pt")
        encoded_sentence = encoded_sentence.to(self.device)
        logits = self.model(**encoded_sentence).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        probability_fluent = probabilities[0][0].item()
        return probability_fluent


# Compute accuracies
def compute_accuracy(df):
    count_correct = 0
    for row_idx in range(len(df)):
        SENTENCE_COL_IDX = 3
        sentence = df.iloc[row_idx, SENTENCE_COL_IDX]

        score = RoBERTaColaFluencyClassifier().score_sentence(sentence)
        print(f"{sentence} {score}")

        LABEL_COL_IDX = 1
        correct_label = df.iloc[row_idx, LABEL_COL_IDX]

        pred = 1 if score > 0.5 else 0
        if pred == correct_label:
            count_correct += 1

    return count_correct / len(df)


print("Accuracy", compute_accuracy(df))
