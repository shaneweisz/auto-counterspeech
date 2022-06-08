from typing import List
from ..metric import Metric
from .gruen_utils import get_gruen
import numpy as np


class GRUEN(Metric):
    @property
    def name(self) -> str:
        return "GRUEN"

    def compute_score(self, predictions: List[str], **kwargs) -> float:
        self.individual_scores = get_gruen(predictions, verbose=kwargs.get("verbose", False))
        self.score = np.mean(self.individual_scores)
        return self.score