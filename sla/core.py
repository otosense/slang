# from sklearn.cluster import KMeans
# from functools import partial

from sla.chunkers import DFLT_CHUNKER
from sla.featurizers import DFLT_FEATURIZER, DFLT_NIPPER


class Snipper:
    """A base class that implements the wf->chk->fv->snip pipeline.
    Default functions for wf_to_chk (a.k.a. chunker), chk_to_fv (a.k.a. featurizer) and fv_to_snip (a.k.a. nipper)
    are given, but usually the user should specify these, and usually these are learned from data.
    """

    def __init__(self, wf_to_chk=DFLT_CHUNKER, chk_to_fv=DFLT_FEATURIZER, fv_to_snip=DFLT_NIPPER):
        self.wf_to_chk = wf_to_chk
        self.chk_to_fv = chk_to_fv
        self.fv_to_snip = fv_to_snip

    def snips_of_wf(self, wf):
        for chk in self.wf_to_chk(wf):
            fv = self.chk_to_fv(chk)
            yield self.fv_to_snip(fv)

    def __call__(self, wf):
        return self.snips_of_wf(wf)
