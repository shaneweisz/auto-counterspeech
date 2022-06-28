import re
from nltk.tokenize import TweetTokenizer
import sys
from pathlib import Path

def clean_str(txt):
    return ' '.join(TweetTokenizer(preserve_case=True).tokenize(txt))


if __name__ == '__main__':
    file_to_clean_path = Path(sys.argv[1])

    cleaned_lines = [clean_str(line) for line in file_to_clean_path.open()]

    cleaned_file_path = file_to_clean_path.parent / (file_to_clean_path.stem + '.cleaned.txt')
    with cleaned_file_path.open('w') as f:
        f.write('\n'.join(cleaned_lines))
