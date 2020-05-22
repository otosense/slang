from collections import defaultdict, Counter
from collections.abc import Mapping
from typing import Union, Callable
from functools import reduce
import matplotlib.pylab as plt

import numpy as np
import pandas as pd

from slang.util import running_mean_gen
from slang.util import snips_to_str as dflt_snips_to_str

from py2store.util import lazyprop, groupby


def _is_a_tuple_of_aligned_iterables(x):
    return hasattr(x, '__len__') and len(x) == 2


def iterate_over_pairs(pairs):
    if _is_a_tuple_of_aligned_iterables(pairs):
        yield from zip(*pairs)
    else:
        yield from pairs  # assume pairs is an iterable of pairs


def mk_snips_of_tag_dict(tags, snips):
    return groupby(zip(tags, snips), lambda x: x[0], lambda x: x[1])


# TODO: Fillna is not additive smoothing, so should make it so?
class TagSnipStats:
    def __init__(self, snips, tags, snips_to_str=None, fillna=0, tag_order=None):
        """

        :param snips: The snips to analyze
        :param tags: The corresponding tags (one per snip)
        :param snips_to_str: A function that outputs a string of characters corresponding to the snip sequence
        :param fillna: The pseudocount that will be used in the probability estimator.
            That is, the value to assign to non-observed (snip, tag) pairs count.
            See https://en.wikipedia.org/wiki/Additive_smoothing
        :param tag_order:
        """
        self.snips = snips
        self.tags = tags
        self.snips_to_str = snips_to_str
        self.fillna = fillna
        self.tag_order = tag_order

    @lazyprop
    def snip_tag_counts(self):
        df = df_of_snip_count_for_tag((self.tags, self.snips),
                                      self.snips_to_str, self.fillna, self.tag_order)
        if self.tag_order is None:
            self.tag_order = df.columns.values
        total_count = df.sum(axis=1)
        snip_order = (total_count.sort_values(ascending=False)).index.values
        df = df.loc[snip_order]
        return df

    @lazyprop
    def snip_count_for_tag(self):
        return tag_snip_count_dict_from_tags_and_snips(self.tags, self.snips)

    @lazyprop
    def snip_order(self):
        return self.snip_tag_counts.index.values

    @lazyprop
    def log_bayes_factor(self):
        return np.log2(bayes_factor_df_from_snip_count_df(self.snip_tag_counts))

    @lazyprop
    def snips_of_tag(self):
        return mk_snips_of_tag_dict(self.tags, self.snips)

    def scores_for_tag(self, tag, snips=None):
        if snips is None:
            snips = self.snips
        return np.array(list(map(self.log_bayes_factor[tag].loc.__getitem__, snips)))

    def plot_snip_count_for_tag(self, snips_to_str=None,
                                figsize=(14, 10), tag_order=None, output_fig=False,
                                ylabel_rotation=90):
        return plot_snip_count_for_tag(self.snip_count_for_tag, snips_to_str=snips_to_str,
                                       figsize=figsize, tag_order=tag_order, output_fig=output_fig,
                                       ylabel_rotation=ylabel_rotation)

    # TODO: Make a decorator that takes care of the `x = x or getattr(self, x, None)` pattern
    def plot_bars_of_tag_snip_stats(self,
                                    figsize=(24, 18), output_fig=False,
                                    ylabel_fontsize=None, ylabel_rotation=90,
                                    tag_order=None, snip_order=None, snips_to_str=None):
        tag_order = tag_order or self.tag_order
        snip_order = snip_order or self.snip_order
        snips_to_str = snips_to_str or self.snips_to_str
        bar_plot_of_tag_snip_stats(self.log_bayes_factor, snips_to_str=snips_to_str, figsize=figsize,
                                   snip_order=snip_order, tag_order=tag_order, output_fig=output_fig,
                                   ylabel_fontsize=ylabel_fontsize, ylabel_rotation=ylabel_rotation)

    def plot_tag_scores(self, tags=None, snips=None, chk_size=1, chk_step=1,
                        figsize=(24, 18), ylabel_fontsize=20, ylabel_rotation=0, predict_thresh=0,
                        normal_style='-k', over_predict_style='ob'):
        if tags is None:
            tags = self.tag_order
        if snips is None:
            snips = np.array(reduce(lambda x, y: x + y, self.snips_of_tag.values(), []))

        gen = tag_scores_gen(self.scores_for_tag, tags, snips, chk_size, chk_step)
        tag_snips_cursor = 0
        n_tags = len(tags)
        plt.figure(figsize=figsize)
        for i, (tag, scores) in enumerate(gen, 1):
            scores = np.array(scores)
            x_vals = np.arange(len(scores))

            plt.subplot(n_tags, 1, i)
            plt.plot(scores, normal_style)
            plt.plot([0, len(scores)], [0, predict_thresh], ':', color='k', alpha=0.8)

            over_predict_thresh = scores >= predict_thresh
            plt.plot(x_vals[over_predict_thresh], scores[over_predict_thresh], over_predict_style)

            #     n_tag_snips = len(tss.snips_of_tag[tag])
            #     these_snip_scores = snip_scores[tag_snips_cursor:(tag_snips_cursor + n_tag_snips)]
            #     tag_snips_idx = list(range(tag_snips_cursor, tag_snips_cursor + len(these_snip_scores)))
            #     plt.plot(tag_snips_idx, these_snip_scores, '-o',  color='k');
            #     tag_snips_cursor += n_tag_snips

            plt.axis('tight')
            plt.ylabel(tag, fontsize=ylabel_fontsize, rotation=ylabel_rotation)

    def plot_smoothed_log_bayes_factors(self, chk_size=1, chk_step=None, tag_order=None):
        return plot_smoothed_log_bayes_factors(self.snips_of_tag, self.log_bayes_factor,
                                               chk_size=chk_size, chk_step=chk_step, tags=tag_order)

    def tags_and_snips_str_gen(self, tags=None, snips=None, snips_to_str=None):
        snips_to_str = snips_to_str or self.snips_to_str or dflt_snips_to_str
        if tags is None and snips is None:
            snips_of_tag = self.snips_of_tag
        else:
            snips_of_tag = mk_snips_of_tag_dict(tags, snips)

        for tag, snips in snips_of_tag.items():
            yield tag, snips_to_str(snips)

    def mk_tags_and_snips_str_string(self, tags=None, snips=None, snips_to_str=None,
                                     tag_snips_format_str="{:<21}: {}\n\n"):
        s = ''
        for tag, snips_str in self.tags_and_snips_str_gen(tags, snips, snips_to_str):
            s += tag_snips_format_str.format(tag, snips_str)
        return s

    def print_tags_and_snips_str(self, tags=None, snips=None, snips_to_str=None,
                                 tag_snips_format_str="{:<21}: {}\n\n"):
        print(self.mk_tags_and_snips_str_string(tags, snips, snips_to_str,
                                                tag_snips_format_str))


class BayesFactors:
    """BayesFactors classifier with sklearn-like interface. predict_probas are the log2 of the bayes factor.
    It is sklearn-like, but to avoid depending on sklearn, it isn't a subclass of BaseEstimator and ClassifierMixin.
    It just has the essentials of the classifier: a `fit`, a `predict_proba` and a derived `predict` method, and
    a `classes_` attribute that indices the columns of the `predict_proba` matrix.
    """

    def __init__(self, pseudocount=0, tag_order=None, alphabet_size=None):
        self.pseudocount = pseudocount
        self.tag_order = tag_order
        self._alphabet_size = alphabet_size  # TODO: Use to tell TagSnipStats it should fill until there

    @lazyprop
    def alphabet_size(self):
        return self._alphabet_size or self.log_bayes_factor_.index.max()

    def fit(self, snips, tags):
        self.tag_snip_stats = TagSnipStats(snips, tags, fillna=self.pseudocount, tag_order=self.tag_order)
        self.log_bayes_factor_ = self.tag_snip_stats.log_bayes_factor.sort_index()
        self.classes_ = self.tag_snip_stats.tag_order
        return self

    @classmethod
    def from_params(cls, log_bayes_factor_, classes_=None):
        self = cls()
        if classes_ is None:
            classes_ = log_bayes_factor_.columns.values
        self.log_bayes_factor_ = log_bayes_factor_[classes_]  # To assert columns contents and ensure order
        self.classes_ = classes_

    def scores_for_tag(self, tag, snips):
        return np.array(list(map(self.log_bayes_factor_[tag].loc.__getitem__, snips)))

    def predict_proba(self, snips):
        scores_for_tag = self.scores_for_tag
        return np.vstack(list(scores_for_tag(tag, snips) for tag in self.classes_)).T

    def predict(self, snips):
        """
        Predict class labels for each snip of snips sequence.
        """
        indices = self.predict_proba(snips).argmax(axis=1)
        return self.classes_[indices]

    def _assert_sanity(self):
        assert set(np.diff(sorted(self.log_bayes_factor_.index))) == {1}, "some snips are missing!"

    _diagnosis = _assert_sanity  # but deprecating _diagnosis

    # TODO: self.snip_to_score([0]) blows up, but self.snip_to_score([0, 1, 2]) gives me a score (should blow up)
    def __call__(self, snip):
        return self.predict_proba([snip])[0]


def mk_model_caller(kind='asis'):
    if kind == 'predict_proba':
        def model_caller(self, snip):
            return self.predict_proba([snip])[0]
    elif kind == 'tag_probs':
        def model_caller(self, snip):
            return {tag: prob for tag, prob in zip(self.classes_, self.predict_proba([snip])[0])}
    elif kind == 'predict':
        def model_caller(self, snip):
            return self.predict([snip])[0]
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return model_caller


class PredictProbaBF(BayesFactors):
    def __call__(self, snip):
        return self.predict_proba([snip])[0]


class TagProbsBF(BayesFactors):
    def __call__(self, snip):
        return {tag: prob for tag, prob in zip(self.classes_, self.predict_proba([snip])[0])}


class PredictBF(BayesFactors):
    def __call__(self, snip):
        return self.predict([snip])[0]


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


def df_of_snip_count_for_tag(snip_count_for_tag, snips_to_str=None, fillna=0, tag_order=None):
    """
    A df representation of snip_count_for_tag
    :param snip_count_for_tag: {tag: {snip: count, ...},...} dict
    :param snips_to_str: A function that transforms snip lists into strings (mapping each snip to a character)
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
    if snips_to_str is not None:
        df = df.reset_index(drop=False)
        df['snip'] = list(snips_to_str(df.index.values))
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


def bar_plot_of_tag_snip_stats(snip_stats_for_tag, snips_to_str=None,
                               figsize=(14, 10), snip_order=None, tag_order=None, output_fig=False,
                               ylabel_fontsize=None, ylabel_rotation=90):
    """
    Multiplot of snip count bars for each tag (in a different row). First row is the total count for each snip.
    :param snip_count_for_tag: {tag: {snip: count, ...},...} nested dict
    :param snips_to_str: A function that transforms snip lists into strings (mapping each snip to a character)
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

    if snips_to_str is None:
        snips_to_str = lambda snips: list(map(str, snips))

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
        if ylabel_fontsize is not None:
            h.set_fontsize(ylabel_fontsize)

        plt.xlabel("")
        if i == 1:
            plt.xticks(list(range(n_snips)), snips_to_str(snip_stats_for_tag.index.values))
            ax.xaxis.tick_top()
        else:
            plt.xticks(list(range(n_snips)), " " * n_snips)

    plt.xticks(list(range(n_snips)), snips_to_str(snip_stats_for_tag.index.values))

    if n_snips < 50:
        for ax in ax_list:
            # ax.grid(True, axis='x')
            ax.grid(True)

    if output_fig:
        return fig


def plot_snip_count_for_tag(snip_count_for_tag, snips_to_str=None,
                            figsize=(14, 10), tag_order=None, output_fig=False,
                            ylabel_fontsize=None, ylabel_rotation=90):
    """
    Multiplot of snip count bars for each tag (in a different row). First row is the total count for each snip.
    :param snip_count_for_tag: {tag: {snip: count, ...},...} nested dict
    :param snips_to_str: A function that transforms snip lists into strings (mapping each snip to a character)
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

    if snips_to_str is None:
        snips_to_str = lambda snips: list(map(str, snips))

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
        if ylabel_fontsize is not None:
            h.set_fontsize(ylabel_fontsize)
        plt.xlabel("")
        if i == 1:
            plt.xticks(list(range(n_snips)), snips_to_str(df.index.values))
            ax.xaxis.tick_top()
        else:
            plt.xticks(list(range(n_snips)), " " * n_snips)

    plt.xticks(list(range(n_snips)), snips_to_str(df.index.values))

    for ax in ax_list:
        ax.grid(True, axis='x')

    if output_fig:
        return fig


def plot_tag_scores_for_snips(snips_of_tag, snip_tag_score_df, tag_order=None,
                              smoothing_window_size=1, figsize=(24, 18),
                              ylabel_fontsize=15, ylabel_rotation=0):
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

        snip_scores = list(running_mean_gen(scores_of_snips(tag, all_snips), smoothing_window_size))
        plt.plot(snip_scores, '-')
        plt.plot([0, len(snip_scores)], [0, 0], ':k')

        n_tag_snips = len(snips_of_tag[tag])
        these_snip_scores = snip_scores[tag_snips_cursor:(tag_snips_cursor + n_tag_snips)]
        tag_snips_idx = list(range(tag_snips_cursor, tag_snips_cursor + len(these_snip_scores)))
        plt.plot(tag_snips_idx, these_snip_scores, 'k-')
        tag_snips_cursor += n_tag_snips

        plt.axis('tight')
        plt.ylabel(tag, fontsize=ylabel_fontsize, rotation=ylabel_rotation)


def tags_and_snips_to_snip_of_tag(tags, snips):
    snips_of_tag = defaultdict(list)

    for tag, snip in zip(tags, snips):
        snips_of_tag[tag].append(snip)

    return dict(snips_of_tag)


# def _tag_order_from_df(df_with_tags_as_columns_and_snips_as_indices):
#     snip_order = df_with_tags_as_columns_and_snips_as_indices.index.values
#     tag_order = df_with_tags_as_columns_and_snips_as_indices.columns.values
#     return snip_order, tag_order


# def scores_of_snips(tag, snips, snip_log_bayes_factor_of_tag):
#     lbf_for_tag = snip_log_bayes_factor_of_tag[tag]
#     if isinstance(snip_log_bayes_factor_of_tag, pd.DataFrame):
#         log_bayes_factor_for_snip = lbf_for_tag.loc.__getitem__
#     elif hasattr(lbf_for_tag, '__getitem__'):
#         log_bayes_factor_for_snip = lbf_for_tag.__getitem__
#     else:
#         assert callable(lbf_for_tag), "At this point lbf_for_tag can only be callable"
#         log_bayes_factor_for_snip = lbf_for_tag
#
#     return list(map(log_bayes_factor_for_snip, snips))


def tag_scores_gen(scores_for_tag, tags, snips, chk_size=1, chk_step=1):
    if isinstance(tags, (str, int, float)):
        tags = [tags]
    for tag in tags:
        yield tag, list(running_mean_gen(scores_for_tag(tag, snips), chk_size, chk_step))


def plot_tag_scores(scores_for_tag, tags, snips, chk_size=1, chk_step=1,
                    figsize=(24, 18), ylabel_fontsize=20, ylabel_rotation=0, predict_thresh=0,
                    normal_style='-k', over_predict_style='ob'):
    gen = tag_scores_gen(scores_for_tag, tags, snips, chk_size, chk_step)
    tag_snips_cursor = 0
    n_tags = len(tags)
    plt.figure(figsize=figsize)
    for i, (tag, scores) in enumerate(gen, 1):
        scores = np.array(scores)
        x_vals = np.arange(len(scores))

        plt.subplot(n_tags, 1, i)
        plt.plot(scores, normal_style)
        plt.plot([0, len(scores)], [0, predict_thresh], ':', color='k', alpha=0.8)

        over_predict_thresh = scores >= predict_thresh
        plt.plot(x_vals[over_predict_thresh], scores[over_predict_thresh], over_predict_style)

        #     n_tag_snips = len(tss.snips_of_tag[tag])
        #     these_snip_scores = snip_scores[tag_snips_cursor:(tag_snips_cursor + n_tag_snips)]
        #     tag_snips_idx = list(range(tag_snips_cursor, tag_snips_cursor + len(these_snip_scores)))
        #     plt.plot(tag_snips_idx, these_snip_scores, '-o',  color='k');
        #     tag_snips_cursor += n_tag_snips

        plt.axis('tight')
        plt.ylabel(tag, fontsize=ylabel_fontsize, rotation=ylabel_rotation)


def scores_of_snips(tag, snips, snip_log_bayes_factor_of_tag):
    return list(map(snip_log_bayes_factor_of_tag[tag].loc.__getitem__, snips))


def plot_smoothed_log_bayes_factors(snips_of_tag, snip_log_bayes_factor_of_tag, chk_size=1, chk_step=1,
                                    tags=None, ylabel_fontsize=15, ylabel_rotation=0):
    # dflt_snip_order, dflt_tag_order = _tag_order_from_df(snip_tag_counts)
    if tags:
        snip_log_bayes_factor_of_tag = snip_log_bayes_factor_of_tag[tags]
    else:
        tags = list(snip_log_bayes_factor_of_tag.columns)
    n_tags = len(tags)

    all_snips = reduce(lambda x, y: x + y, snips_of_tag.values(), [])

    plt.figure(figsize=(24, 18))

    tag_snips_cursor = 0
    for i, tag in enumerate(tags, 1):
        plt.subplot(n_tags, 1, i)
        snip_scores = list(
            running_mean_gen(scores_of_snips(tag, all_snips, snip_log_bayes_factor_of_tag), chk_size, chk_step))
        plt.plot(snip_scores, '-')
        plt.plot([0, len(snip_scores)], [0, 0], ':k')

        n_tag_snips = len(snips_of_tag[tag])
        these_snip_scores = snip_scores[tag_snips_cursor:(tag_snips_cursor + n_tag_snips)]
        tag_snips_idx = list(range(tag_snips_cursor, tag_snips_cursor + len(these_snip_scores)))
        plt.plot(tag_snips_idx, these_snip_scores, 'k-')
        tag_snips_cursor += n_tag_snips

        plt.axis('tight')
        plt.ylabel(tag, fontsize=ylabel_fontsize, rotation=ylabel_rotation)


def snip_scores_from_lookup(snips, snip_to_score):
    if isinstance(snip_to_score, (pd.Series, dict)):
        snip_to_score = snip_to_score.__getitem__
    elif isinstance(snip_to_score, pd.DataFrame):
        _snip_to_score = {k: snip_to_score[k].loc.__getitem__ for k in list(snip_to_score.columns)}
        snip_to_score = lambda snip: {k: lookup(snip) for k, lookup in _snip_to_score}
    return map(snip_to_score, snips)

    # assert isinstance(snip_to_score, pd.DataFrame), \
    #     "snip_to_score needs to be DataFrame whose index values are snips and columns are the different score kinds " \
    #     "you want to compute"
