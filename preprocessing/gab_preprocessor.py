from pathlib import Path
from typing import List, Any
import pandas as pd
import ast
import re
from .base_preprocessor import Preprocessor


class GabPreprocessor(Preprocessor):
    def extract_rows(self, file_path: Path) -> List[List[Any]]:
        df = pd.read_csv(file_path)
        df = self.remove_conversations_with_no_hate_speech(df)
        hs_cs_pairs = []
        for i in range(len(df)):
            row = df.iloc[i]

            text = row["text"]
            hs_idxes = ast.literal_eval(row["hate_speech_idx"])
            cs_responses = ast.literal_eval(row["response"])

            hss = []
            for hs_idx in hs_idxes:
                r = rf"{hs_idx}\..*"
                try:
                    hs = re.findall(r, text)[0]
                    hs = hs.split(".", maxsplit=1)[
                        1
                    ].strip()  # get HS text after the index
                    hss.append(hs)
                except IndexError:
                    print(
                        f"Warning: HS idx {hs_idx} is not a valid index "
                        + "in the conversation. Skipping this entry."
                    )
            for hs in hss:
                for cs in cs_responses:
                    hs_cs_pairs.append([hs, cs])

        return hs_cs_pairs

    @staticmethod
    def remove_conversations_with_no_hate_speech(df: pd.DataFrame) -> pd.DataFrame:
        return df[pd.notnull(df["hate_speech_idx"])]
