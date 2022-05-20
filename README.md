![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# Automating Counterspeech in Dialogue Systems

## Project Overview

Counterspeech is a direct response to hate speech that seeks to undermine it. A key advantage of using counterspeech to combat hate speech is that it does not violate freedom of speech (compared to measures like content moderation and blocking users). However, manual generation of good counterspeech is time-consuming and expensive. AI, therefore, could have a powerful impact in improving the *scalability* of applying counterspeech. However, research on AI approaches to generating counterspeech is still in its infancy. As such, this project thus aims to contribute towards improved automatic generation of counterspeech.

## Requirements

The code is based on Python 3.8. Please install the main dependencies as below:
```
pip install -r requirements.txt
```

Optionally install development dependencies with:
```
pip install -r requirements-dev.txt
```

## Data

### Preprocessing

Firstly, notice that the unprocessed counterspeech datasets (as released by their authors) are located in `data/unprocessed`.

After preprocessing the datasets, the resulting files are located at the top-level of the `data` directory.

The preprocessing can be replicated by running:

```bash
cd data
python preprocessing/preprocess.py -i unprocessed/<data>.csv [-o <output_file_path>]
```

### Analysis

Data analysis can be conducted on all datasets by running:

```bash
cd data
source analysis/analyse_all_datasets.sh
```

or on an individual dataset by running:

```bash
cd data
python analysis/analyse_dataset.py -f <data.csv>
```
