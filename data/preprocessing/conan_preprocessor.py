from pathlib import Path
import pandas as pd
from typing import List, Tuple
from base_preprocessor import Preprocessor


class ConanPreprocessor(Preprocessor):
    def extract_hs_cs_pairs(self, file_path: Path) -> List[Tuple[str, str]]:
        df = pd.read_csv(file_path)
        hs_cs_pairs = []
        for _, row in df.iterrows():
            hs = row["hateSpeech"]
            cs = row["counterSpeech"]
            hs_cs_pairs.append((hs, cs))
        return hs_cs_pairs


class MultiTargetConanPreprocessor(Preprocessor):
    def extract_hs_cs_pairs(self, file_path: Path) -> List[Tuple[str, str]]:
        df = pd.read_csv(file_path)
        hs_cs_pairs = []
        for _, row in df.iterrows():
            hs = row["HATE_SPEECH"]
            cs = row["COUNTER_NARRATIVE"]
            hs_cs_pairs.append((hs, cs))
        return hs_cs_pairs
