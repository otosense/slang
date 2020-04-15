# from sklearn.cluster import KMeans
# from functools import partial

from warnings import warn
from typing import Callable, Any, Optional
from slang.stypes import Waveform, Chunk, Chunker, Featurizer, Quantizer, Snip, Snips, FVs
from slang.chunkers import DFLT_CHUNKER
from slang.featurizers import DFLT_FEATURIZER, DFLT_QUANTIZER

WfCallback = Optional[Callable[[Waveform], Any]]


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

    def snips_of_wf(self, wf: Waveform) -> Snips:
        warn("The name 'snips_of_wf' be replaced by 'wf_to_snips' soon.")
        for chk in self.wf_to_chks(wf):
            fv = self.chk_to_fv(chk)
            yield self.fv_to_snip(fv)

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
        return self.snips_of_wf(wf)


class KvDataSource:
    def __init__(self, kv_store, key_to_tag=None):
        self.kv_store = kv_store
        self.key_to_tag = key_to_tag

    def key_wf_gen(self):
        return iter(self.kv_store.items())

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
