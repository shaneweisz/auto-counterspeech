![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Automating Counterspeech in Dialogue Systems

## Project Overview

*Counterspeech* is a direct response to hate speech that seeks to undermine it and challenge the hate narratives. The key advantage of using counterspeech to combat hate speech, as opposed to measures like content moderation and blocking users, is that it does not violate freedom of speech. However, manual generation of good counterspeech is time-consuming and expensive. AI, therefore, could have a powerful impact in improving the *scalability* of applying counterspeech. However, research on AI approaches to generating counterspeech is still in its infancy, and has yet to be approached from a general dialogue systems framing.

This project thus aims to contribute towards improved automatic generation of counterspeech using dialogue systems, along with investigating the impact on the general conversational ability of such a system. In particular, our primary modelling approach is based on fine-tuning [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium#:~:text=DialoGPT%20is%20a%20SOTA%20large,single%2Dturn%20conversation%20Turing%20test) on the [MultiCONAN](https://github.com/marcoguerini/CONAN#Multitarget-CONAN) dataset, a dataset comprising a set of hate speech inputs and appropriate [counterspeech](https://dangerousspeech.org/counterspeech/) responses produced under the supervision of trained NGO operators from [Stop Hate UK](https://www.stophateuk.org/).

You can interact with one of the trained counterspeech-enhanced systems via our [public web demo](https://huggingface.co/spaces/shaneweisz/AutoCounterspeech).

## Requirements

The recommended python version is python 3.8+. You can check your python version is at least 3.8 by running `python --version`

We recommend that you create a top-level virtual environment with:

```bash
python -m venv .venv
```

Activate the virtual environment with:
```bash
source .venv/bin/activate
```

Upgrade your pip:
```bash
pip install --upgrade pip
```

Then install the main dependencies as below (it may take a while):

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

The preprocessing can be replicated by running, for example:

```bash
python preprocess.py -i data/unprocessed/Multitarget-CONAN.csv -o data/preprocessed/REPLICATED-Multitarget-CONAN.csv
```

### Train-Val-Test Split

To replicate the multi-conan train-val-test split, run:

```bash
python split_multiconan.py -i data/preprocessed/Multitarget-CONAN.csv -o data/splits/REPLICATED-multitarget-conan
```

To replicate the gab or reddit train-val-test split, run:
```bash
python split_gab.py -i data/preprocessed/gab.csv -o data/splits/REPLICATED-gab
```

### Exploratory Data Analysis (EDA)

Data analysis can be conducted on all preprocessed datasets (gab, reddit, conan, multitarget-conan) by running, for example:

```bash
python eda.py -o data/eda/REPLICATED-EDA.txt
```

or on individual dataset(s) by running, for example:

```bash
python eda.py -f data/splits/multitarget-conan/train.csv data/splits/multitarget-conan/val.csv  data/splits/multitarget-conan/test.csv -o data/eda/REPLICATED-MultiCONAN-EDA.txt
```

## Fine-tuning

To fine-tune DialoGPT on MultiCONAN:
```
python train.py -d data/splits/multitarget-conan -c config/train.config.mc.json
```

This should output a trained model at `models/DialoGPT-finetuned-multiCONAN`.

Make sure you have a GPU, otherwise it will likely be very slow. With a GPU, training should not take more than an hour.

## Interacting

Interact with base DialoGPT by running:
```
python interact.py -m microsoft/DialoGPT-medium
```

Similarly, interact with a trained fine-tuned model by running, for example:
```
python interact.py -m shaneweisz/DialoGPT-finetuned-gab-multiCONAN --config_overrides "num_beams=3;no_repeat_ngram_size=5"
```

Note that this pulls the model `shaneweisz/DialoGPT-finetuned-gab-multiCONAN` from Hugging Face. You could use your own local trained model instead.
## Decoding

Generate responses to a text file of inputs by running, for example:

```
python decode.py --model shaneweisz/DialoGPT-finetuned-gab-multiCONAN --config config/decode.config.json -i data/splits/gab/test.inputs.txt -o TEST-DECODING-FOLDER
```

## Evaluation

Counterspeech response predictions can be evaluated with respect to inputs and gold-standard references through various metrics by running, for example:

```bash
python evaluate.py -r data/splits/gab/test.references.txt -p TEST-DECODING-FOLDER/predictions.txt -i data/splits/gab/test.inputs.txt
```

Metrics can be controlled using the `-m` flag. The supported metrics are:

1. Relevance:

    * `bleu1`, `bleu2`, `bleu3`, `bleu4`
    * `rouge1`, `rouge2`, `rougeL`
    * `bertscore`

2. Diversity:

    * `distinct1`, `distinct2`
    * `entropy1`, `entropy2`, `entropy3`, `entropy4`

3. Fluency:

    * `fluency`

4. Response lengths:

    * `avg-length`, `max-length`, `min-length`, `median-length`

5. Toxicity:

    * `toxicity`

To use BLEU, the nltk tokenizer requires the punkt package. You can install this locally using the python interpreter as follows:

```bash
python -c 'import nltk; nltk.download("punkt")'
```

## Reproducing Experiment Results

All experiments were conducted on Nvidia Ampere (A100) GPU nodes through the Cambridge HPC via the Slurm Workload Manager.

All slurm scripts are found in the `slurm_scripts` folder. Inspect and adapt these scripts to configure your own experiments or training details.

### Model training

The experiments use three fine-tuned models: `models/DialoGPT-finetuned-multiCONAN`, `models/DialoGPT-finetuned-gab`, and `models/DialoGPT-finetuned-gab`

The respective `slurm.train` scripts can be run to reproduce the training of each the respective fine-tuned counterspeech models. For example, after running `sbatch slurm.train.mc.wilkes3`, the trained model from fine-tuning DialoGPT on MultiCONAN will be found at `models/DialoGPT-finetuned-multiCONAN`.

### Experiments

The various experiment results can then be reproduced by running the respective `slurm.exp` scripts.

There were two main counterspeech experiments using these models: `Main` (comparing fine-tuned models to baselines evaluated using MultiCONAN test set) and `Gab` (comparing models, but using the Gab test set). The results can be reproduced by running `sbatch slurm_scripts/slurm.exp.main.wilkes3` and `sbatch slurm_scripts/slurm.exp.gab.wilkes3` respectively.

Finally, run `sbatch slurm_scripts/slurm.exp.conv.wilkes3` to reproduce the general conversational ability experiment results.
