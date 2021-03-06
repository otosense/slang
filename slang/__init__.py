from slang.core import KvDataSource, Snipper

from slang.util import snips_to_str as dflt_snips_to_str
from slang.util import ddir

from slang.chunkers import (
    fixed_step_chunker,
    DFLT_CHUNKER,
    DFLT_CHK_SIZE,
    mk_chunker,
)

from slang.featurizers import (
    DFLT_WF_TO_SPECTR,
    DFLT_FEATURIZER,
    DFLT_QUANTIZER,
    tile_fft
)

from slang.spectrop import (
    SpectralProjector,
    mk_chk_fft,
    geo_mat_11,
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
