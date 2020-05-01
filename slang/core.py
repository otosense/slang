# from sklearn.cluster import KMeans
# from functools import partial

from warnings import warn
from typing import Callable, Any, Optional
from slang.stypes import Waveform, Chunk, Chunker, Featurizer, Quantizer, Snip, Snips, FVs
from slang.chunkers import DFLT_CHUNKER
from slang.featurizers import DFLT_FEATURIZER, DFLT_QUANTIZER

WfCallback = Optional[Callable[[Waveform], Any]]


class TagWfStore:
    def __init__(self, wf_store, key_to_tag=None, key_filt=None):
        self.wf_store = wf_store
        self.key_to_tag = key_to_tag
        self.key_filt = key_filt

    def key_wf_gen(self):
        pass

    def wfs(self):
        pass

    def tag_wfs(self):
        pass


from py2store.util import lazyprop


class WfSource:
    def __init__(self, wfs, key_filt=None):
        self.wfs = wfs
        self.key_filt = key_filt

    @lazyprop
    def keys(self):
        return tuple(filter(self.key_filt, self.wfs))

    def wf_gen(self):
        for k in self.keys:
            yield self.wfs[k]

    def key_wf_gen(self):
        for k in self.keys:
            yield k, self.wfs[k]


class AnnotedWfSource(WfSource):
    def __init__(self, wfs, annots=None, key_filt=None):
        super().__init__(wfs, key_filt)
        self.annots = annots

    @lazyprop
    def keys(self):
        annots_keys = set(self.annots)
        return tuple([k for k in super().keys() if k in annots_keys])

    def annots_gen(self):
        for k in self.keys:
            yield self.annots[k]

    def annot_wf_gen(self):
        for k in self.keys:
            yield self.annots[k], self.wfs[k]

    def key_annot_wf_gen(self):
        for k in self.keys:
            yield self.annots[k], self.wfs[k]


# Notes:
"""
- key_to_tag not general enough because tag not general enough. Sometimes it's not a categorical.
    Sometimes it's multiple. Sometimes we have context information that needs to be associated with the annot. 
"""
class KvDataSource:
    def __init__(self, kv_store, key_to_tag=None, key_filt=None):
        self.kv_store = kv_store
        self.key_to_tag = key_to_tag
        self.key_filt = key_filt

    def key_wf_gen(self):
        kv_store = self.kv_store
        for k in filter(self.key_filt, self.kv_store):
            yield k, kv_store[k]

    def key_tag_wf_gen(self, wf_callback: WfCallback = None, iterate_over_wf_callaback_output=False):
        assert self.key_to_tag is not None, "You need to have a key_to_tag function to do that!"
        if wf_callback is None:
            for k, wf in self.key_wf_gen():
                yield k, self.key_to_tag(k), wf
        else:
            assert callable(wf_callback), "wf_callback needs to be callable."

            if iterate_over_wf_callaback_output:
                for k, wf in self.key_wf_gen():
                    tag = self.key_to_tag(k)
                    for x in wf_callback(wf):
                        yield k, tag, x
            else:
                for k, wf in self.key_wf_gen():
                    yield k, self.key_to_tag(k), wf_callback(wf)

    def key_tag_chks_gen(self, wf_to_chk):
        yield from self.key_tag_wf_gen(wf_callback=wf_to_chk, iterate_over_wf_callaback_output=True)

    def key_tag_fvs_gen(self, wf_to_chk, chk_to_fv):
        wf_to_fv = lambda wf: list(map(chk_to_fv, wf_to_chk(wf)))
        yield from self.key_tag_wf_gen(wf_callback=wf_to_fv, iterate_over_wf_callaback_output=True)

    def key_tag_snips_gen(self, wf_to_chk, chk_to_fv, fv_to_snip):
        def wf_to_snips(wf):
            return list(fv_to_snip(chk_to_fv(chk)) for chk in wf_to_chk(wf))

        yield from self.key_tag_wf_gen(wf_callback=wf_to_snips, iterate_over_wf_callaback_output=True)

    def key_chks_gen(self, wf_to_chk):
        for k, wf in self.key_wf_gen():
            for chk in wf_to_chk:
                yield k, chk

    def key_fvs_gen(self, wf_to_chk, chk_to_fv):
        for k, chk in self.key_chks_gen(wf_to_chk):
            yield k, chk_to_fv(chk)

    def key_snips_gen(self, wf_to_chk, chk_to_fv, fv_to_snip):
        for k, fv in self.key_fvs_gen(wf_to_chk, chk_to_fv):
            yield k, fv_to_snip(fv)

    def chk_tag_pairs(self, wf_to_chks):
        return ((chk, tag) for _, tag, chk in self.key_tag_chks_gen(wf_to_chks))

    def fv_tag_pairs(self, wf_to_chks, chk_to_fv):
        return ((fv, tag) for _, tag, fv in self.key_tag_fvs_gen(wf_to_chks, chk_to_fv))


class Snipper:
    """A base class that implements the wf->chk->fv->snip pipeline.
    Default functions for wf_to_chk (a.k.a. chunker), chk_to_fv (a.k.a. featurizer) and fv_to_snip (a.k.a. nipper)
    are given, but usually the user should specify these, and usually these are learned from data.
    """

    def __init__(self,
                 wf_to_chks: Chunker = DFLT_CHUNKER,
                 chk_to_fv: Featurizer = DFLT_FEATURIZER,
                 fv_to_snip: Quantizer = DFLT_QUANTIZER):
        self.wf_to_chks = wf_to_chks
        self.chk_to_fv = chk_to_fv
        self.fv_to_snip = fv_to_snip

    def wf_to_snips(self, wf: Waveform) -> Snips:
        # warn("The name 'snips_of_wf' be replaced by 'wf_to_snips' soon.")
        for chk in self.wf_to_chks(wf):
            fv = self.chk_to_fv(chk)
            yield self.fv_to_snip(fv)

    snips_of_wf = wf_to_snips  # alias for back-compatibility

    def wf_to_fvs(self, wf: Waveform) -> FVs:
        for chk in self.wf_to_chks(wf):
            yield self.chk_to_fv(chk)

    def chk_to_snip(self, chk: Chunk) -> Snip:
        return self.fv_to_snip(self.chk_to_fv(chk))

    def wf_to_snips(self, wf: Waveform) -> Snips:
        for chk in self.wf_to_chks(wf):
            fv = self.chk_to_fv(chk)
            yield self.fv_to_snip(fv)

    def __call__(self, wf: Waveform) -> Snips:
        return self.wf_to_snips(wf)
