from pathlib import Path
from typing import List
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    OUTPUT_HEADER = "hate_speech,counter_speech"

    def __init__(self, input_file_path: Path, output_file_path: Path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def preprocess(self):
        print(f"Extracting HS-CS pairs from {self.input_file_path}")
        hs_cs_pairs = self.extract_rows(self.input_file_path)

        print(f"Writing HS-CS pairs to {self.output_file_path}")
        self._write_to_csv(hs_cs_pairs, self.output_file_path)

    @abstractmethod
    def extract_rows(self, file_path: Path) -> List[List[any]]:
        """
        Implementation differs depends on the input file format and the desired columns
        to extract from the rows.
        """
        pass

    def _write_to_csv(self, rows: List[List[any]], output_file_path: Path) -> Path:
        with open(output_file_path, "w") as f:
            f.write(self.OUTPUT_HEADER + "\n")
            for row in rows:
                for i, col in enumerate(row):
                    if type(col) == str:
                        # escape quotes in strings to avoid errors when writing to csv
                        col = col.replace('"', '""').replace("\n", " ")
                        row[i] = f'"{col}"'
                f.write(",".join(row) + "\n")
