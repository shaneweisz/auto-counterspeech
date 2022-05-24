import re
from .metric import Metric
from .bleu import BLEU


class MetricFactory:
    @staticmethod
    def from_metric_name(metric_name: str) -> Metric:
        if metric_name.lower().startswith("bleu"):
            N = int(re.search(r"\d+", metric_name).group())
            return BLEU(N=N)
        else:
            UNSUPPORTED_METRIC_ERR_MSG = f"Unsupported metric: `{metric_name}`"
            raise ValueError(UNSUPPORTED_METRIC_ERR_MSG)
