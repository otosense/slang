from matplotlib.pylab import *
from numpy import *
from collections import Counter
import pandas as pd
from itertools import chain
from slang.core import DFLT_CHUNKER
# from agen.sound.audio import Sound
from slang.featurizers import _rms_and_crossing_levels, _zero_crossing_ratio
import seaborn as sns


def zero_crossing_count(wf):
    return sum(diff(array(wf) > 0).astype(int))


def heatmap_of_rms_and_crossing_levels(wf_gen, chunker=DFLT_CHUNKER):
    x = list()
    y = list()
    for wf in wf_gen:
        for chk in chunker(wf):
            xx, yy = _rms_and_crossing_levels(chk)
            x.append(xx)
            y.append(yy)

    t = pd.DataFrame({'rms': x, 'zero_crossings': y}).groupby(['rms', 'zero_crossings']).size()
    # t.plot(x='vol', y='zero_crossings', kind='scatter', figsize=(9, 9), alpha=0.2)
    t = t.unstack()
    figure(figsize=(11, 9));
    sns.heatmap(t.T, cmap='gray_r');


def hist_of_zero_crossings(wf_gen, chunker=DFLT_CHUNKER):
    c = Counter(zero_crossing_count(chk) for chk in chain(*map(chunker, wf_gen)))

    pd.Series(c).sort_index().plot(kind='bar', figsize=(16, 5))
    plt.xticks(range(max(c.keys())), range(max(c.keys())))
    plt.xlabel('zero crossings');
    plt.ylabel('count')


def rms_zcr_scatter(wf_gen, chunker=DFLT_CHUNKER):
    def gen(wf_gen):
        for wf in wf_gen:
            for chk in chunker(wf):
                yield std(chk), _zero_crossing_ratio(chk)

    t = pd.DataFrame([{'rms': rms, 'zero_crossing_ratio': zcr} for rms, zcr in gen(wf_gen)])
    t.plot(x='rms', y='zero_crossing_ratio', kind='scatter', figsize=(9, 9), alpha=0.2)
