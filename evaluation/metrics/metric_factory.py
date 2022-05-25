import re

from .base_metric import Metric
from .relevance import BLEU, ROUGE, BERTScore
from .diversity import DistinctN, EntropyN
from .fluency import GRUEN


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
        else:
            err_msg = f"Unsupported metric: `{metric_name}`"
            raise ValueError(err_msg)
