from pathlib import Path
from typing import List, Tuple
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def preprocess(self):
        print(f"Extracting HS-CS pairs from {self.file_path}")
        hs_cs_pairs = self.extract_hs_cs_pairs(self.file_path)

        new_file_path = self.write_to_new_csv(hs_cs_pairs)
        print(f"Processed file written to {new_file_path}")

    @abstractmethod
    def extract_hs_cs_pairs(self, file_path: Path) -> List[Tuple[str, str]]:
        pass

    def write_to_new_csv(self, hs_cs_pairs: List[Tuple[str, str]]) -> Path:
        new_file_path = Path(__file__).parent / (self.file_path.stem + "_processed.csv")
        with open(new_file_path, "w") as f:
            HEADER = "HateSpeech,CounterSpeech"
            f.write(HEADER + "\n")
            for hs, cs in hs_cs_pairs:
                f.write(f'"{hs}","{cs}"\n')
        return new_file_path
