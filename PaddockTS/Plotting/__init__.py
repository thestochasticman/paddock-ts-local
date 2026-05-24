"""Plotting subpackage.

Importing this package bumps matplotlib's default font sizes so every
plot the pipeline produces (phenology, climate panels, terrain, etc.)
has readable ticks, axis labels, and titles after the PDF report's
~3× shrink onto an A4 page.

Override with `plt.rcParams.update(...)` at the call site if you need
different sizing for a one-off figure. Only affects matplotlib — the
calendar plot is a PIL composite and configures its own fonts.
"""

import matplotlib as _mpl

_mpl.rcParams.update({
    'font.size':       16,   # baseline; other elements inherit from this
    'axes.titlesize':  20,   # ax.set_title(...)
    'figure.titlesize': 22,  # fig.suptitle(...)
    'axes.labelsize':  18,   # ax.set_xlabel / ax.set_ylabel
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16,
})
