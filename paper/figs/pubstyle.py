"""One typographic and colour style for every data figure in the paper.

TWO RULES, and both exist because breaking them produced real defects:

1. AUTHOR AT THE WIDTH YOU PLACE AT. A figure drawn 6.9in wide and included
   at \\linewidth inside a single CAS column (3.33in) is scaled by 0.48, so an
   8pt label reaches the reader at under 4pt -- below the ~7pt floor a
   reviewer can read in print. Use COL for a figure in a `figure` and WIDE for
   one in a `figure*`, and never rescale in LaTeX. The sizes set here are then
   the sizes on the page.

2. A GIVEN ARM IS A GIVEN COLOUR EVERYWHERE. The colours live in ARM/MARK
   below, keyed by intervention, so the prior is the same vermillion in the
   attention figure as in the envelopes. The palette is Okabe-Ito, which stays
   distinguishable under all three common colour-vision deficiencies -- and,
   since the marker shapes in MARK differ too, the figures survive being
   printed in greyscale.

Shape follows panel count, not preference: a two-panel comparison whose x
axis is shared reads side by side and belongs in a `figure*`; a single panel,
or an image grid, belongs in one column.
"""
import matplotlib.pyplot as plt

# CAS double-column geometry, in inches.
COL = 3.33    # one column   -> \begin{figure}
WIDE = 6.86   # both columns -> \begin{figure*}

# Okabe-Ito, colour-vision-deficiency safe.
OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "purple": "#CC79A7", "grey": "#999999",
      "sky": "#56B4E9", "yellow": "#F0E442"}
INK, MUTED, RULE = "#1a1a1a", "#555555", "#666666"

# Fixed assignment: an arm keeps its colour and marker across all figures.
ARM = {
    "baseline":     OI["grey"],
    "prior":        OI["verm"],
    "simclr":       OI["blue"],
    "simsiam":      OI["sky"],
    "dino":         OI["purple"],
    "augmentation": OI["orange"],
    "transfer":     OI["green"],
}
MARK = {
    "baseline": "o", "prior": "^", "simclr": "s", "simsiam": "v",
    "dino": "D", "augmentation": "P", "transfer": "X",
}

# Nothing on a page goes below SMALL; that is the legibility floor the
# re-authoring pass was for, and it is enforced here rather than remembered.
SMALL = 6.5     # in-axes annotations
TICK = 7.0
LABEL = 7.5
TITLE = 8.0
LEGEND = 6.8


def use():
    """Apply the shared rcParams. Call once at the top of a figure script."""
    plt.rcParams.update({
        "font.size": LABEL,
        "axes.labelsize": LABEL,
        "axes.titlesize": TITLE,
        "legend.fontsize": LEGEND,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.4,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#333333",
        # Type 42 embeds the fonts as TrueType rather than Type 3 bitmapped
        # outlines, which is what Elsevier's preflight asks for.
        "pdf.fonttype": 42,
    })


def panel(ax, title=None):
    """Left-aligned panel title, the convention used by every panel here."""
    if title:
        ax.set_title(title, fontsize=TITLE, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax
