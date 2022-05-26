from typing import List
from ..base_metric import Metric
from .gruen_utils import get_gruen
import numpy as np


class GRUEN(Metric):
    @property
    def name(self) -> str:
        return "GRUEN"

    def compute_score(self, predictions: List[str], **kwargs) -> float:
        scores = get_gruen(predictions, verbose=kwargs.get("verbose", False))
        score = np.mean(scores)
        return score
