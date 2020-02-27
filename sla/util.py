from numpy import array, nan, empty, arange, unique, ones
from sla.util_data import displayable_unichr

####### Seeing snips

unichr_code_of_snip = array(displayable_unichr
                            + list(unique(list(set(range(33, 20000)).difference(displayable_unichr)))))
snip_of_unichr_code = (nan * ones(unichr_code_of_snip.max() + 1)).astype(int)
snip_of_unichr_code[unichr_code_of_snip] = arange(len(unichr_code_of_snip))

# assert that snip_of_unichr_code is the inverse of unichr_code_of_snip
assert (all(snip_of_unichr_code[unichr_code_of_snip] == arange(len(unichr_code_of_snip))))


def snip_to_str(snip):
    return chr(unichr_code_of_snip[snip])


def str_of_snips(snips):
    return ''.join(map(chr, unichr_code_of_snip[snips]))


def snips_of_str(snips):
    return array(snip_of_unichr_code[list(map(ord, snips))])
