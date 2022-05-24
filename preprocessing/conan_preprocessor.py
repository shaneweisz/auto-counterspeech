from pathlib import Path
import pandas as pd
from typing import List, Tuple
from .base_preprocessor import Preprocessor


class ConanPreprocessor(Preprocessor):
    def extract_hs_cs_pairs(self, file_path: Path) -> List[Tuple[str, str]]:
        df = pd.read_csv(file_path)
        hs_cs_pairs = []
        for _, row in df.iterrows():
            cn_id = row["cn_id"]
            if self._is_english(cn_id):
                hs = row["hateSpeech"]
                cs = row["counterSpeech"]
                hs_cs_pairs.append((hs, cs))
        return hs_cs_pairs

    @staticmethod
    def _is_english(cn_id):
        """
        cn_id: counter narrative id
            Example: "ENT1ST0001HS0033CN000021" (originally english)
                  or "ITT1ST0023HS0055CN001068T1" (translated to english from italian)
        """
        originally_english = cn_id[:2] == "EN"
        translated_to_english = cn_id[-2:] == "T1"
        return originally_english or translated_to_english
