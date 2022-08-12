
## General Conversational Ability Framework

This is mainly based on the code and experiments from Microsoft's DialoGPT paper. [[code]](https://github.com/microsoft/DialoGPT) [[paper]](https://arxiv.org/abs/1911.00536)

All the necessary code and files are in the `general_conv_reddit` folder.

### Setup

#### Conda environment

We use the conda environment from the DialoGPT repo. Ensure you deactivate the `auto-counterspeech` venv before switching to this conda environment, by running `deactivate`.

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
python evaluate.py --refs_dir <path_to_refs_dir> --hyp_file <path_to_predictions.cleaned.txt>
```

Note: `clean-str.py` tokenizes a set of predictions into a cleaned format expected by the `evaluate.py` script.
