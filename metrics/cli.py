import re
from typing import List

from .metric import Metric
from .relevance import BLEU, ROUGE, BERTScore
from .diversity import DistinctN, EntropyN
from .fluency import GRUEN, RoBERTaColaFluencyClassifier
from .response_length import ResponseLengthSummaryStatistic
from .toxicity import UnbiasedToxicRoBERTa


def load_metrics(metric_names: List[str]) -> List[Metric]:
    return [MetricFactory.from_metric_name(name) for name in metric_names]


class MetricFactory:
    @staticmethod
    def from_metric_name(metric_name: str) -> Metric:
        # Relevance metrics
        if metric_name.lower().startswith("bleu"):
            N = int(re.search(r"\d+", metric_name).group())
            if N > 4:
                err_msg = f"BLEU N must be <= 4, got {N}"
                raise ValueError(err_msg)
            return BLEU(N=N)
        elif metric_name.startswith("rouge"):
            valid_rouge_types = ["rouge1", "rouge2", "rougeL"]
            if metric_name not in valid_rouge_types:
                err_msg = f"Unsupported rouge_type: {metric_name}."
                err_msg += "\nSupported rouge_types: " + ", ".join(valid_rouge_types)
                raise ValueError(err_msg)
            return ROUGE(rouge_type=metric_name)
        elif metric_name.lower().startswith("bert"):
            return BERTScore()
        # Diversity metrics
        elif metric_name.lower().startswith("dist"):
            N = int(re.search(r"\d+", metric_name).group())
            if N > 2:
                err_msg = f"Dist-N must be <= 2, got {N}"
                raise ValueError(err_msg)
            return DistinctN(N=N)
        elif metric_name.lower().startswith("ent"):
            N = int(re.search(r"\d+", metric_name).group())
            if N > 4:
                err_msg = f"Ent-N must be <= 4, got {N}"
                raise ValueError(err_msg)
            return EntropyN(N=N)
        # Fluency metrics
        elif metric_name.lower().startswith("gruen"):
            return GRUEN()
        elif metric_name.lower().startswith("roberta-cola"):
            return RoBERTaColaFluencyClassifier()
        # Response length metrics
        elif metric_name.lower().startswith("max-len"):
            return ResponseLengthSummaryStatistic("Max")
        elif metric_name.lower().startswith("min-len"):
            return ResponseLengthSummaryStatistic("Min")
        elif metric_name.lower().startswith("avg-len"):
            return ResponseLengthSummaryStatistic("Avg")
        elif metric_name.lower().startswith("median-len"):
            return ResponseLengthSummaryStatistic("Median")
        # Toxicity metrics
        elif metric_name.lower().startswith("toxic"):
            return UnbiasedToxicRoBERTa()
        else:
            err_msg = f"Unsupported metric: `{metric_name}`"
            raise ValueError(err_msg)
