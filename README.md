![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# Automating Counterspeech in Dialogue Systems

## Project Overview

"Counterspeech" is a direct response to hate speech that seeks to undermine it. The key advantage of using counterspeech to combat hate speech, as opposed to measures like content moderation and blocking users, is that it does not violate freedom of speech. However, manual generation of good counterspeech is time-consuming and expensive. AI, therefore, could have a powerful impact in improving the *scalability* of applying counterspeech. However, research on AI approaches to generating counterspeech is still in its infancy. As such, this project thus aims to contribute towards improved automatic generation of counterspeech.

## Requirements

The recommended python version is python 3.8+. You can check your python version is at least 3.8 by running `python --version`

We recommend that you create a top-level virtual environment with:

```bash
python -m venv .venv
```

Then install the main dependencies as below:

```bash
pip install -r requirements.txt
```

Optionally install development dependencies with:

```bash
pip install -r requirements-dev.txt
```

The nltk tokenizer requires the punkt package. This is used for the BLEU evaluation metrics.

You can install this locally using the python interpreter as follows:

```bash
python -c 'import nltk; nltk.download("punkt")'
```

## Data

### Preprocessing

Firstly, notice that the original counterspeech datasets (as released by their authors) are located in `data/originals`.

After preprocessing the datasets, the resulting files are located at the top-level of the `data` directory.

The preprocessing can be replicated by running:

```bash
python preprocess.py -i data/originals/<data>.csv [-o <output_file_path>]
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
python analysis/analyse_dataset.py -f <data>.csv
```

## Evaluation

A file containing model predictions can be evaluated as follows:

```bash
python evaluate.py -f <preds>.csv [-m <metrics>]
```

Note that the predictions file must have the following fields: `input`, `prediction` and `reference`.

The supported metrics are:

* `bleu1`, `bleu2`, `bleu3`, `bleu4`
* `rouge1`, `rouge2`, `rougeL`
* `bertscore`
