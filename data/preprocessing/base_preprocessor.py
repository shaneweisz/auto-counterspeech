from pathlib import Path
from typing import List, Tuple
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    OUTPUT_HEADER = "hate_speech,counter_speech"

    def __init__(self, input_file_path: Path, output_file_path: Path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def preprocess(self):
        print(f"Extracting HS-CS pairs from {self.input_file_path}")
        hs_cs_pairs = self.extract_hs_cs_pairs(self.input_file_path)

        print(f"Writing HS-CS pairs to {self.output_file_path}")
        self._write_to_csv(hs_cs_pairs, self.output_file_path)

    @abstractmethod
    def extract_hs_cs_pairs(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        This function must be implemented by child classes. Depending on the specific
        input file format, the list of hate speech-counterspeech (HS-CS) pairs will be
        extracted differently.
        """
        pass

    def _write_to_csv(
        self, hs_cs_pairs: List[Tuple[str, str]], output_file_path: Path
    ) -> Path:
        with open(output_file_path, "w") as f:
            f.write(Preprocessor.OUTPUT_HEADER + "\n")
            for hs, cs in hs_cs_pairs:
                # escape quotes in hs and cs to avoid errors when writing to csv
                hs = hs.replace('"', '""')
                cs = cs.replace('"', '""')

                f.write(f'"{hs}","{cs}"\n')
