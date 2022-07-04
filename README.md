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

Install development dependencies with:

```bash
pip install -r requirements-dev.txt
```

## Data

### Preprocessing

Firstly, notice that the original counterspeech datasets (as released by their authors) are located in `data/unprocessed`.

After preprocessing the datasets, the resulting files are located in the `data/preprocessed` directory.

The preprocessing can be replicated by running:

```bash
python preprocess.py -i data/unprocessed/<data>.csv [-o <output_file_path>]
```

### Train-Val-Test Split

To replicate the multi-conan train-val-test split, run:

```bash
python split_multiconan.py -i <path_to_multiconan_csv> -o <output_dir>
```

To replicate the gab or reddit train-val-test split, run:
```bash
python split_gab.py -i <path_to_gab_csv> -o <output_dir>
```

### Exploratory Data Analysis (EDA)

Data analysis can be conducted on all datasets (gab, reddit, conan, multitarget-conan) by running:

```bash
python eda.py [-o <output_file_path>]
```

or on an individual dataset by running:

```bash
python eda.py -f <data>.csv [-o <output_file_path>]
```

## Generating Counterspeech Responses

Counterspeech response predictions using a model (e.g. `dialoGPT`) can be made on a set of inputs as follows:

```
python decode.py --model <modelname> --config <config>.json -i <inputs>.txt [-o <predictions>.txt]
```

## Evaluation

Counterspeech response predictions can be evaluated with respect to inputs and gold-standard references through various metrics by running:

```bash
python evaluate.py -r <references.txt> -p <predictions.txt> -i <inputs.txt> [-m <metrics>] [-v --verbose]
```

The supported metrics are:

1. Relevance:

    * `bleu1`, `bleu2`, `bleu3`, `bleu4`
    * `rouge1`, `rouge2`, `rougeL`
    * `bertscore`

2. Diversity:

    * `distinct1`, `distinct2`
    * `entropy1`, `entropy2`, `entropy3`, `entropy4`

3. Fluency:

    * `gruen`, `fluency`

4. Response lengths:

    * `avg-length`, `max-length`, `min-length`, `median-length`

5. Toxicity:

    * `toxicity`

Note that to use GRUEN, the following steps are necessary:

* Download the pretrained CoLA classifier [here](https://drive.google.com/file/d/1Hw5na_Iy4-kGEoX60bD8vXYeJDQrzyj6/view?usp=sharing) and unzip it in the `metrics/fluency` directory.
* Run `python -m spacy download en_core_web_md` to download `en_core_web_md` from the `spacy` module.

To use BLEU, the nltk tokenizer requires the punkt package. You can install this locally using the python interpreter as follows:

```bash
python -c 'import nltk; nltk.download("punkt")'
```

## General Conversational Ability Experiments

This is mainly based on the code and experiments from Microsoft's DialoGPT paper. [[code]](https://github.com/microsoft/DialoGPT) [[paper]](https://arxiv.org/abs/1911.00536)

All the necessary code and files are in the `general_conv_reddit` folder:

### Setup

#### Conda environment

We use the conda environment from the DialoGPT repo. Ensure you deactivate the `auto-counterspeech` venv before switching to this conda environment.

```bash
cd general_conv_reddit
```

Create the conda environment:

```bash
conda env create -f LSP-linux.yml -n LSP
```

Activate the conda environment:
```bash
conda activate LSP
```

Add pandas:
```bash
conda install pandas
```

#### Install 3rd-party metric evaluation scripts:

```bash
cd general_conv_reddit/metrics
```

METEOR (requires `java`):
```bash
wget http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xf meteor-1.5.tar.gz
```

NIST (requires `perl` and `cpan`):
```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/mteval-v14.pl
cpan install XML:Twig Sort:Naturally String:Util
```

### Evaluation

To evaluate a set of predictions:

```
cd general_conv_reddit
python util/clean-str.py path_to_predictions.txt
python evaluate.py --refs_dir path_to_refs_dir --hyp_file path_to_predictions.cleaned.txt
```

Note: `clean-str.py` tokenizes a set of predictions into a cleaned format expected by the `evaluate.py` script.
