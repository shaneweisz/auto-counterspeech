import re
from .metric import Metric
from .bleu import BLEU


class MetricFactory:
    @staticmethod
    def from_metric_name(metric_name: str) -> Metric:
        if metric_name.lower().startswith("bleu"):
            N = int(re.search(r"\d+", metric_name).group())
            if N > 4:
                err_msg = f"BLEU N must be <= 4, got {N}"
                raise ValueError(err_msg)
            return BLEU(N=N)
        else:
            err_msg = f"Unsupported metric: `{metric_name}`"
            raise ValueError(err_msg)
