from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute_score(
        self,
        predictions: List[str],
        references: List[str] = None,
        inputs: List[str] = None,
    ) -> float:
        pass
