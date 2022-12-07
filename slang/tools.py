"""Various slang tools"""
from functools import singledispatch
from collections.abc import Sequence
from collections import Counter

DFLT_PRIOR_COUNT = 1


@singledispatch
def snip_probs(count_of_snip: dict, prior_count=DFLT_PRIOR_COUNT):
    total_count = sum(count_of_snip.values()) + len(count_of_snip) * prior_count
    return {
        snip: (count_of_snip + prior_count) / float(total_count)
        for snip, count_of_snip in count_of_snip.items()
    }


@snip_probs.register
def _(count_of_snip: Sequence, prior_count=DFLT_PRIOR_COUNT):
    return snip_probs(Counter(count_of_snip), prior_count=prior_count)
