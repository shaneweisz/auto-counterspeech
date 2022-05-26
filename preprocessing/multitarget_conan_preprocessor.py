from pathlib import Path
import pandas as pd
from typing import Any, List
from .base_preprocessor import Preprocessor


class MultiTargetConanPreprocessor(Preprocessor):
    OUTPUT_HEADER = "hate_speech,counter_speech,target"

    def extract_rows(self, file_path: Path) -> List[List[Any]]:
        df = pd.read_csv(file_path)
        extracted_rows = []
        for _, row in df.iterrows():
            hs = row["HATE_SPEECH"]
            cs = row["COUNTER_NARRATIVE"]
            target = row["TARGET"]
            extracted_rows.append([hs, cs, target])
        return extracted_rows
