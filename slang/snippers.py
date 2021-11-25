"""Snipping: Feature vector quantization"""

from typing import Callable
import itertools
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

from slang.chunkers import fixed_step_chunker
from slang.snip_stats import BayesFactors
from slang.core import Snipper
from slang.util import row_euclidean_distance, mk_callable


class DfltWfToChk:
    def __init__(self, chk_size=2048, chk_step=None):
        if chk_step is None:
            chk_step = chk_size
        self.chk_size = chk_size
        self.chk_step = chk_step

    def fit(self, *args, **kwargs):
        return self  # no fitting required

    def __call__(self, wf):
        yield from fixed_step_chunker(
            wf, chk_size=self.chk_size, chk_step=self.chk_step
        )


@mk_callable('single_transform')
class PcaChkToFv(PCA):
    def __init__(self, n_components=5, **kwargs):
        super().__init__(n_components=n_components, **kwargs)

    # def __call__(self, fv):
    #     return self.transform([fv])[0]


@mk_callable('single_transform')
class LdaChkToFv(LinearDiscriminantAnalysis):
    def __init__(self, n_components=5, **kwargs):
        super().__init__(n_components=n_components, **kwargs)


DfltChkToFv = PcaChkToFv


class FvToSnip:  # TODO: Mixin? ABC?
    # fit: optional
    # __call__: required
    # fvs_to_snips: required?
    # fvs_to_snip_distance_pairs: required?
    @property
    def fv_of_snip(self):
        """array providing representative fv for each snip"""
        raise NotImplemented(
            'Not implemented -- should be implemented in subclass of FvToSnipMixin'
        )
        # return None


from slang.stypes import FV, FVs


# TODO: choose n_clusters from len(X) at fit time
# TODO: post_fit_proc shouldn't be here, but external and optional
@mk_callable('single_predict')
class KMeansFvToSnip(KMeans, FvToSnip):
    stats_of_snip = None  # will be filled when instance is fit

    # cluster_centers_ = None  # will be filled when instance is fit
    # n_clusters = None
    #
    # fv_of_snip = cluster_centers_  # Snipper's api expectation
    # alphabet_size = n_clusters  # Snipper's api expectation

    @property
    def fv_of_snip(self):
        return self.cluster_centers_

    @property
    def alphabet_size(self):
        return self.n_clusters

    def __init__(self, n_clusters=47, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)

    # def __call__(self, fv: FV):
    #     return self.predict([fv])[0]

    def fit(self, fvs: FVs, y=None, sample_weight=None):
        super().fit(fvs, y, sample_weight)
        # self.post_fit_proc(fvs, y, sample_weight=sample_weight)
        return self

    # fvs_to_snips = KMeans.predict  # alias for predict
    # ... replaced (because need to handle iterator of fvs) by:
    def fvs_to_snips(self, fvs, sample_weight=None):
        # return self.predict(list(fvs), sample_weight=sample_weight)
        if isinstance(fvs, (list, tuple, np.ndarray)):
            return self.predict(fvs, sample_weight=sample_weight)
        else:
            return map(lambda x: self.predict([x]), fvs)

    def fvs_to_snip_distance_pairs(self, fvs: FVs):
        """iterator of (snip, distance_to_snip_centroid) pairs"""
        fvs = np.array(list(fvs))
        snips = self.fvs_to_snips(fvs)
        return zip(snips, row_euclidean_distance(fvs, self.fv_of_snip[snips]))

    # def post_fit_proc(self, fvs, y=None, **kwargs):
    #     d = defaultdict(list)
    #     # TODO: Better than a for loop? (d.update(self.snip_and_distance(fvs)) doesn't work, but want something like))
    #     for snip, distance in self.fvs_to_snip_distance_pairs(fvs):
    #         d[snip].append(distance)
    #
    #     # TODO: A numpy array might be more useful here
    #     # self.stats_of_snip = np.empty(self.alphabet_size) * np.nan
    #
    #     self.stats_of_snip = {snip:
    #                               dict(count=len(distances),
    #                                    dist_sum=sum(distances),
    #                                    dist_sum2=sum(np.array(distances) ** 2))
    #                           for snip, distances in d.items()}
    #
    #     return self


class KMeansFvToSnipDist(KMeansFvToSnip):
    def __call__(self, fv: FV):
        return list(self.fvs_to_snip_distance_pairs([fv]))[0]


DfltFvToSnip = KMeansFvToSnip

from collections.abc import Iterable


def is_iterable(x):
    """Similar in nature to :func:`callable`, ``is_iterable`` returns
    ``True`` if an object is `iterable`_, ``False`` if not.
    >>> is_iterable([])
    True
    >>> is_iterable(1)
    False"""
    return isinstance(x, Iterable)


def _assure_pair(iterables):
    """ """
    if hasattr(iterables, '__len__'):
        if len(iterables) == 2:
            return iterables
        elif len(iterables) == 1 and is_iterable(iterables[0]):
            return list(zip(*iterables[0]))
        # TODO: else?... raise? If None on purpose, say it and why!
    else:
        return list(zip(*iterables))


_get_pairs = _assure_pair  # but deprecating


def _assure_zipped(iterables):
    """ """
    if hasattr(iterables, '__len__'):
        if len(iterables) == 1:
            return iterables
        else:
            return zip(*iterables)
    else:
        return zip(*iterables)


def _is_a_tuple_of_aligned_iterables(x):
    return hasattr(x, '__len__') and len(x) == 2


def iterate_over_pairs(pairs):
    if _is_a_tuple_of_aligned_iterables(pairs):
        yield from zip(*pairs)
    else:
        yield from pairs  # assume pairs is an iterable of pairs


class FittableSnipper(Snipper):
    wf_to_chks: Callable
    chk_to_fv: Callable  # TODO: Make a "Callable and Fittable" type
    fv_to_snip: FvToSnip  # TODO: Make a "Callable and Fittable" type

    dflt_cls_of_name: dict = {
        'wf_to_chks': DfltWfToChk,
        'chk_to_fv': PcaChkToFv,
        'fv_to_snip': DfltFvToSnip,
    }

    @classmethod
    def mk_component(cls, obj, role=None, assert_func=None):
        if isinstance(obj, type):
            obj = obj()
        elif obj is None:
            obj = cls.dflt_cls_of_name.get(role)()
        elif not callable(obj):
            assert isinstance(obj, dict)
            obj = cls.dflt_cls_of_name.get(role)(**obj)
        else:
            pass  # as is
        if assert_func is not None:
            assert assert_func(obj)
        return obj

    def __init__(self, wf_to_chks=None, chk_to_fv=None, fv_to_snip=None):

        wf_to_chks = self.mk_component(wf_to_chks, 'wf_to_chks', assert_func=callable)
        chk_to_fv = self.mk_component(chk_to_fv, 'chk_to_fv', assert_func=callable)
        fv_to_snip = self.mk_component(fv_to_snip, 'fv_to_snip', assert_func=callable)

        super().__init__(wf_to_chks, chk_to_fv, fv_to_snip)

        # TODO: Find a more concise way to take care of block above. Like... (but not working)
        # _locals = locals()
        # for obj_role in ['wf_to_chks', 'chk_to_fv', 'fv_to_snip', 'snip_to_score']:
        #     _locals[obj_role] = self.mk_component(_locals[obj_role], obj_role, assert_func=callable)

    # TODO: Make the next three methods more DRY (see also ClassificationSnipper.fit_snip_to_score)
    def fit_wf_to_chks(self, *wfs_tags):
        if hasattr(self.wf_to_chks, 'fit'):
            chks, tags = _assure_pair(
                wfs_tags
            )  # need to generalize to situations with no tags
            self.wf_to_chks.fit(chks, tags)
        return self

    def fit_chk_to_fv(self, *chks_tags):
        if hasattr(self.chk_to_fv, 'fit'):
            chks, tags = _assure_pair(chks_tags)
            self.chk_to_fv.fit(chks, tags)
        return self

    def fit_fv_to_snip(self, *fvs_tags):
        if hasattr(self.fv_to_snip, 'fit'):
            fvs, tags = _assure_pair(fvs_tags)
            self.fv_to_snip.fit(fvs, tags)
        return self

    # TODO: Dagify all this. Too hardcoded. No caching. No flexibility

    def chk_tag_gen(self, wf_tag_gen):
        for wf, tag in wf_tag_gen():
            for chk in self.wf_to_chks(wf):
                yield chk, tag

    def fv_tag_gen(self, wf_tag_gen):
        for chk, tag in self.chk_tag_gen(wf_tag_gen):
            yield self.chk_to_fv(chk), tag

    def snip_tag_gen(self, wf_tag_gen):
        for fv, tag in self.fv_tag_gen(wf_tag_gen):
            yield self.fv_to_snip(fv), tag

    def fit_pipeline(self, wf_tags_gen):
        self.fit_wf_to_chks(wf_tags_gen())
        self.fit_chk_to_fv(self.chk_tag_gen(wf_tags_gen))
        self.fit_fv_to_snip(self.fv_tag_gen(wf_tags_gen))

    def fit(self, X, y=None):
        if y is None:
            wf_tags_gen = lambda: itertools.product(X, [y])
        else:
            wf_tags_gen = lambda: zip(X, y)
        return self.fit_pipeline(wf_tags_gen)


class ClassificationSnipper(FittableSnipper):
    snip_to_score: Callable  # TODO: Make a "Callable and Fittable" type

    dflt_cls_of_name: dict = dict(
        FittableSnipper.dflt_cls_of_name, snip_to_score=BayesFactors
    )

    def __init__(
        self,
        wf_to_chks=DfltWfToChk(),
        chk_to_fv=DfltChkToFv,
        fv_to_snip=DfltFvToSnip,
        snip_to_score=BayesFactors,
    ):
        super().__init__(wf_to_chks, chk_to_fv, fv_to_snip)

        snip_to_score = self.mk_component(
            snip_to_score, 'snip_to_score', assert_func=callable
        )
        self.snip_to_score = snip_to_score

    def fit_snip_to_score(self, *snips_tags):
        if hasattr(self.snip_to_score, 'fit'):
            snips, tags = _assure_pair(snips_tags)
            self.snip_to_score.fit(snips, tags)
        return self

    # TODO: Dagify all this. Too hardcoded. No caching. No flexibility

    def fit_pipeline(self, wf_tags_gen):
        super().fit_pipeline(wf_tags_gen)
        self.fit_snip_to_score(self.snip_tag_gen(wf_tags_gen))


# from slang.snip_stats import ClassifiedMomentsFitter
# from itertools import chain
#
#
# class OutlierSnipper(FittableSnipper):
#     snip_to_score: Callable  # TODO: Make a "Callable and Fittable" type
#
#     dflt_cls_of_name: dict = dict(FittableSnipper.dflt_cls_of_name, snip_to_score=ClassifiedMomentsFitter)
#
#     def __init__(self, wf_to_chks=DfltWfToChk(),
#                  chk_to_fv=DfltChkToFv,
#                  fv_to_snip=DfltFvToSnip,
#                  snip_to_score=ClassifiedMomentsFitter):
#         super().__init__(wf_to_chks, chk_to_fv, fv_to_snip)
#
#         snip_to_score = self.mk_component(snip_to_score, 'snip_to_score', assert_func=callable)
#         self.snip_to_score = snip_to_score
#
#     def fit_snip_to_score(self, *dist_snips):
#         if hasattr(self.snip_to_score, 'fit'):
#             dist, snips = _assure_pair(dist_snips)
#             self.snip_to_score.fit(dist, snips)
#         return self
#
#     # TODO: Dagify all this. Too hardcoded. No caching. No flexibility
#
#     def fit_pipeline(self, wf_gen):
#         # super().fit_pipeline(wf_tags_gen)
#         self.fit_wf_to_chks(wf_gen())
#         self.fit_chk_to_fv(self.chk_tag_gen(wf_tags_gen))
#         fvs = map(self.chk_to_fv, chain.from_iterable(map(self.wf_to_chks,
#         self.fit_fv_to_snip(self.fv_tag_gen(wf_tags_gen))
#
#         self.fit_snip_to_score(self.snip_tag_gen(wf_tags_gen))

# def snips_of_wf(self, wf: Waveform) -> Snips:
#     warn("The name 'snips_of_wf' be replaced by 'wf_to_snips' soon.")
#     for chk in self.wf_to_chks(wf):
#         fv = self.chk_to_fv(chk)
#         yield self.fv_to_snip(fv)
#
# def wf_to_fvs(self, wf: Waveform) -> FVs:
#     for chk in self.wf_to_chks(wf):
#         yield self.chk_to_fv(chk)
#
# def chk_to_snip(self, chk: Chunk) -> Snip:
#     return self.fv_to_snip(self.chk_to_fv(chk))
#
# def wf_to_snips(self, wf: Waveform) -> Snips:
#     for chk in self.wf_to_chks(wf):
#         fv = self.chk_to_fv(chk)
#         yield self.fv_to_snip(fv)


SlangClassifier = ClassificationSnipper  # alias for back compatibility
