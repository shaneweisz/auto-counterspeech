from pathlib import Path
from typing import List


def read_list_from_file(file_path: Path) -> List[str]:
    with open(file_path) as f:
        return [line.strip() for line in f]


def write_text_to_file(file_path: Path, text: str, append: bool = False):
    mode = "a" if append else "w"
    with open(file_path, mode) as f:
        f.write(text)


def write_list_to_file(file_path: Path, lines: List[str], append: bool = False):
    text = "\n".join(lines)
    write_text_to_file(file_path, text, append)
