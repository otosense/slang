from collections import defaultdict, Counter
from functools import reduce
from collections import deque
import matplotlib.pylab as plt

import numpy as np

import pandas as pd


def running_mean(it, chk_size=2):
    """
    Running mean (moving average) on iterator.
    Note: When input it is list-like, ut.stats.smooth.sliders version of running_mean is 4 times more efficient with
    big (but not too big, because happens in RAM) inputs.
    :param it: iterable
    :param chk_size: width of the window to take means from
    :return:
    """
    it = iter(it)
    if chk_size > 1:
        c = 0
        fifo = deque([], maxlen=chk_size)
        for i, x in enumerate(it, 1):
            fifo.append(x)
            c += x
            if i >= chk_size:
                break

        yield c / chk_size

        for x in it:
            c += x - fifo[0]  # NOTE: seems faster than fifo.popleft
            fifo.append(x)
            yield c / chk_size

    else:
        for x in it:
            yield x


def tag_slice_iter_from_slices_of_tag_dict(slices_of_tag):
    """
    Get an iterator of (tag, (bt, tt)) pairs
    :param slices_of_tag: a {tag: [(bt, tt),...], ...} dict listing slices annotated by tags
    :return: a tag, (bt, tt) iterator
    """
    for tag, slices in slices_of_tag.items():
        for sl in slices:
            yield tag, sl


# def snip_count_for_tag_from_wf_and_tag_slice_iter(wf, tag_slice_iter, snips_of_wf, sr=None):
#     """
#
#     :param wf: waveform of oto.sound.audio.Sound object
#     :param tag_slice_iter:
#     :param snips_of_wf:
#     :param sr:
#     :return:
#     """
#
#     # from slang.utils.audio_core import Sound
#     snip_count_for_tag = defaultdict(Counter)
#
#     for tag, sl in tag_slice_iter:
#         chk = sound[slice(*sl)].wf  # wf chunk for sl slice
#         chk_snips = snips_of_wf(chk)
#         snip_count_for_tag[tag].update(chk_snips)
#
#     return dict(snip_count_for_tag)

def tag_snip_count_dict_from_tags_and_snips(tags, snips):
    snip_count_for_tag = defaultdict(dict)

    for (tag, snip), count in Counter(zip(tags, snips)).items():
        snip_count_for_tag[tag][snip] = count

    return dict(snip_count_for_tag)


def df_of_snip_count_for_tag(snip_count_for_tag, string_of_snips=None, fillna=0, tag_order=None):
    """
    A df representation of snip_count_for_tag
    :param snip_count_for_tag: {tag: {snip: count, ...},...} dict
    :param string_of_snips: A function that transforms snip lists into strings (mapping each snip to a character)
    :param fillna: What to fill missing values with
    :param tag_order: Serves both to specify an order of the tags, and to specify a subset of tags if we don't want all
    :return: A dataframe of snip (in rows) counts for each tag (in columns)
    """
    if isinstance(snip_count_for_tag, tuple) and len(snip_count_for_tag) == 2:
        snip_count_for_tag = tag_snip_count_dict_from_tags_and_snips(*snip_count_for_tag)

    df = pd.DataFrame(snip_count_for_tag).fillna(fillna)
    if tag_order is not None:
        df = df[tag_order]
    df.index.names = ['snip']
    if string_of_snips is not None:
        df = df.reset_index(drop=False)
        df['snip'] = list(string_of_snips(df.index.values))
        df = df.set_index('snip')
    return df


def snip_order_from_snip_count_df(snip_count_df):
    total_count = snip_count_df.sum(axis=1)
    return (total_count.sort_values(ascending=False)).index.values


def bayes_factor_df_from_snip_count_df(snip_count_df):
    smoothed_count = snip_count_df + 1
    total_snip_count = smoothed_count.sum(axis=1)
    prob_given_tag = smoothed_count / smoothed_count.sum(axis=0)
    # prob_given_not_tag = total_snip_count.sub(smoothed_count, level=0, fill_value=0, axis=0)
    prob_given_not_tag = -smoothed_count.sub(total_snip_count, axis=0)
    prob_given_not_tag /= prob_given_not_tag.sum(axis=0)
    return prob_given_tag / prob_given_not_tag


def log_bayes_factor_bayes_factor_df_from_snip_count_df(snip_count_df):
    return np.log2(bayes_factor_df_from_snip_count_df(snip_count_df))


def bar_plot_of_tag_snip_stats(snip_stats_for_tag, string_of_snips=None,
                               figsize=(14, 10), snip_order=None, tag_order=None, output_fig=False,
                               ylabel_rotation=90):
    """
    Multiplot of snip count bars for each tag (in a different row). First row is the total count for each snip.
    :param snip_count_for_tag: {tag: {snip: count, ...},...} nested dict
    :param string_of_snips: A function that transforms snip lists into strings (mapping each snip to a character)
    :param figsize:
    :param output_fig:
    :param tag_order: Serves both to specify an order of the tags, and to specify a subset of tags if we don't want all
    :param ylabel_rotation: Will be applied to the ylabel
    :return:
    """
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.gcf()

    if string_of_snips is None:
        string_of_snips = lambda snips: list(map(str, snips))

    if not isinstance(snip_stats_for_tag, pd.DataFrame):
        snip_stats_for_tag = pd.DataFrame(snip_stats_for_tag)

    if snip_order is not None:
        snip_stats_for_tag = snip_stats_for_tag.loc[snip_order, :]

    if tag_order is not None:
        snip_stats_for_tag = snip_stats_for_tag[tag_order]

    n_tags = len(snip_stats_for_tag.columns)
    n_snips = len(snip_stats_for_tag)

    ax_list = list()
    for i, tag in enumerate(snip_stats_for_tag.columns, 1):
        sr = snip_stats_for_tag[tag]
        ax = plt.subplot(n_tags + 1, 1, i)
        ax_list.append(ax)
        positive_lidx = sr > 0
        sr.plot(kind='bar', color=positive_lidx.map({True: 'b', False: '#D3D3D3'}))
        h = plt.ylabel(tag)
        h.set_rotation(ylabel_rotation)
        plt.xlabel("")
        if i == 1:
            plt.xticks(list(range(n_snips)), string_of_snips(snip_stats_for_tag.index.values))
            ax.xaxis.tick_top()
        else:
            plt.xticks(list(range(n_snips)), " " * n_snips)

    plt.xticks(list(range(n_snips)), string_of_snips(snip_stats_for_tag.index.values))

    if n_snips < 50:
        for ax in ax_list:
            # ax.grid(True, axis='x')
            ax.grid(True)

    if output_fig:
        return fig


def plot_snip_count_for_tag(snip_count_for_tag, string_of_snips=None,
                            figsize=(14, 10), tag_order=None, output_fig=False, ylabel_rotation=90):
    """
    Multiplot of snip count bars for each tag (in a different row). First row is the total count for each snip.
    :param snip_count_for_tag: {tag: {snip: count, ...},...} nested dict
    :param string_of_snips: A function that transforms snip lists into strings (mapping each snip to a character)
    :param figsize:
    :param output_fig:
    :param tag_order: Serves both to specify an order of the tags, and to specify a subset of tags if we don't want all
    :param ylabel_rotation: Will be applied to the ylabel
    :return:
    """
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.gcf()

    if string_of_snips is None:
        string_of_snips = lambda snips: list(map(str, snips))

    df = df_of_snip_count_for_tag(snip_count_for_tag, fillna=0, tag_order=tag_order)
    n_tags = len(df.columns)
    df['ALL'] = df.sum(axis=1)
    df = df.sort_values('ALL', ascending=False)
    n_snips = len(df)

    ax_list = list()
    for i, tag in enumerate(df.columns, 1):
        sr = df[tag]
        ax = plt.subplot(n_tags + 1, 1, i)
        ax_list.append(ax)
        sr.plot(kind='bar')
        h = plt.ylabel(tag)
        h.set_rotation(ylabel_rotation)
        plt.xlabel("")
        if i == 1:
            plt.xticks(list(range(n_snips)), string_of_snips(df.index.values))
            ax.xaxis.tick_top()
        else:
            plt.xticks(list(range(n_snips)), " " * n_snips)

    plt.xticks(list(range(n_snips)), string_of_snips(df.index.values))

    for ax in ax_list:
        ax.grid(True, axis='x')

    if output_fig:
        return fig


def plot_tag_scores_for_snips(snips_of_tag, snip_tag_score_df, tag_order=None,
                              smoothing_window_size=1, figsize=(24, 18),
                              ylabel_font_size=15, ylabel_rotation=0):
    assert isinstance(snip_tag_score_df, pd.DataFrame), "isinstance(snip_tag_score_df, pd.DataFrame)"

    def scores_of_snips(snips, tag):
        return list(map(snip_tag_score_df[tag].loc.__getitem__, snips))

    if tag_order is None:
        tag_order = list(snips_of_tag.keys())
    n_tags = len(tag_order)

    all_snips = reduce(lambda x, y: x + y, snips_of_tag.values(), [])

    plt.figure(figsize=figsize)

    tag_snips_cursor = 0
    for i, tag in enumerate(tag_order, 1):
        plt.subplot(n_tags, 1, i)

        snip_scores = list(running_mean(scores_of_snips(tag, all_snips), smoothing_window_size))
        plt.plot(snip_scores, '-')
        plt.plot([0, len(snip_scores)], [0, 0], ':k')

        n_tag_snips = len(snips_of_tag[tag])
        these_snip_scores = snip_scores[tag_snips_cursor:(tag_snips_cursor + n_tag_snips)]
        tag_snips_idx = list(range(tag_snips_cursor, tag_snips_cursor + len(these_snip_scores)))
        plt.plot(tag_snips_idx, these_snip_scores, 'k-')
        tag_snips_cursor += n_tag_snips

        plt.axis('tight')
        plt.ylabel(tag, fontsize=ylabel_font_size, rotation=ylabel_rotation)

#
# def tag_order_from_snip_tag_counts(snip_tag_counts):
#     # TODO: Sort
#     snip_order = snip_order_from_snip_count_df(snip_tag_counts)
#     tag_order = list(snip_tag_counts['tag'].unique())
#     return snip_order, tag_order
#
#
# from ut.util.uiter import running_mean
# from functools import reduce
#
# smoothing_window_size_chk = 10
#
# snip_order, tag_order = tag_order_from_snip_tag_counts(snip_tag_counts)
# n_tags = len(tag_order)
#
# all_snips = reduce(lambda x, y: x + y, snips_of_tag.values(), [])
#
# plt.figure(figsize=(24, 18));
#
# tag_snips_cursor = 0
# for i, tag in enumerate(tag_order, 1):
#     plt.subplot(n_tags, 1, i);
#     snip_scores = list(running_mean(scores_of_snips(tag, all_snips, log_bayes_factor), smoothing_window_size_chk))
#     plt.plot(snip_scores, '-');
#     plt.plot([0, len(snip_scores)], [0, 0], ':k')
#
#     n_tag_snips = len(snips_of_tag[tag])
#     these_snip_scores = snip_scores[tag_snips_cursor:(tag_snips_cursor + n_tag_snips)]
#     tag_snips_idx = list(range(tag_snips_cursor, tag_snips_cursor + len(these_snip_scores)))
#     plt.plot(tag_snips_idx, these_snip_scores, 'k-');
#     tag_snips_cursor += n_tag_snips
#
#     plt.axis('tight')
#     plt.ylabel(tag, fontsize=15, rotation=0);
#
#
