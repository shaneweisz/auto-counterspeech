from pathlib import Path
import matplotlib.pyplot as plt


DROPBOX_IMAGES_PATH = Path("/Users/shaneweisz/Dropbox/Apps/Overleaf/MPhil Thesis Automating Counterspeech/Figures")


tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    # Use 12pt font in plots, to match 12pt font in document
    "axes.labelsize": 10,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def savefig_to_dropbox(filename):
    full_filename = f"{DROPBOX_IMAGES_PATH}/{filename}"
    plt.savefig(full_filename, bbox_inches="tight")


def set_tex_font_params():
    plt.rcParams.update(tex_fonts)


set_tex_font_params()

FIG_WIDTH = 4.2
FIG_HEIGHT = 3.5
FIGSIZE = (FIG_WIDTH, FIG_HEIGHT)
