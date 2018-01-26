import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
from numpy import array
from itertools import cycle

# Plot parameter sets
allParamSets = {}

# Journal defaults
journal_figsize = (16,12)
journal_textsize = 28
journal_markersize = 16
journal_markeredgewidth = journal_markersize/4
journal_linewidth = 4
journalRCParams = {}
journalRCParams['lines.linewidth'] = journal_linewidth
journalRCParams['lines.markersize'] = journal_markersize
journalRCParams['lines.markeredgewidth'] = journal_markeredgewidth
journalRCParams['legend.fontsize'] = journal_textsize
journalRCParams['font.size'] = journal_textsize
journalRCParams['figure.figsize'] = journal_figsize
journalRCParams['savefig.bbox'] = 'tight'
journalRCParams['image.cmap'] = 'inferno'
journalRCParams['legend.loc'] = 'lower left'
journalRCParams['xtick.labelsize'] = 'large'
journalRCParams['ytick.labelsize'] = 'large'
journalRCParams['axes.labelsize'] = 'large'
allParamSets['journal'] = journalRCParams 

def blackLinesGenerator():
    return cycle(["k-","k--","k-.","k:"])
    
def blackMarkersAndLinesGenerator():
    return cycle(["k+","k:","k--","k.","k-"])

def setPlotDefaults(kind):
    for key, value in allParamSets[kind].items():
        matplotlib.rcParams[key] = value

def removeBorder(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
