"""Signal transformers
Tools to transform signals into a structured language.
"""
from slang.core import KvDataSource, Snipper

from slang.util import (
    snips_to_str as dflt_snips_to_str,
    str_to_snips as dflt_str_to_snips,
)
from slang.util import ddir

from slang.chunkers import (
    fixed_step_chunker,
    DFLT_CHUNKER,
    DFLT_CHK_SIZE,
    mk_chunker,
    fannout_chunker,
)

from slang.featurizers import (
    DFLT_WF_TO_SPECTR,
    DFLT_FEATURIZER,
    DFLT_QUANTIZER,
    tile_fft,
    mk_wf_to_spectr,
)

from slang.spectrop import (
    SpectralProjector,
    mk_chk_fft,
    geo_mat_11,
    logarithmic_bands_matrix,
)

from slang.snippers import (
    DfltWfToChk,
    PcaChkToFv,
    DfltChkToFv,
    FittableSnipper,
    ClassificationSnipper,
    FvToSnip,
    KMeansFvToSnip,
    KMeansFvToSnipDist,
    DfltFvToSnip,
)

from slang.settings import SLANG_DATA_DIR
from slang.stores import local_snips_store
