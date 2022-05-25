import re

from .base_metric import Metric
from .bleu import BLEU
from .rouge import ROUGE
from .bert_score import BERTScore
from .distinct_n import DistinctN


class MetricFactory:
    @staticmethod
    def from_metric_name(metric_name: str) -> Metric:
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
        elif metric_name.startswith("bert"):
            return BERTScore()
        elif metric_name.startswith("distinct"):
            N = int(re.search(r"\d+", metric_name).group())
            return DistinctN(N=N)
        else:
            err_msg = f"Unsupported metric: `{metric_name}`"
            raise ValueError(err_msg)
