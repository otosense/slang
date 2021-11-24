"""Types"""

from typing import Callable, Union, Any, List, Tuple
from numpy import ndarray, int16, int32, float32, float64

# Note I'm using Tuple to denote fixed sized sequences and List to denote a sliceable unbounded sized iterator and
# tuple as a fixed

from typing import TypeVar, Optional, Iterable, Sequence
from numbers import Number

from atypes import MyType

from atypes import (
    FixedSizeSeq,
    VarSizeSeq,
    Key,
    Sample,
    Waveform,
    Waveforms,
    KeyWfGen,
    Chunk,
    Chunks,
    Chunker,
    Feature,
    FV,
    FVs,
    Featurizer,
    ChkFeaturizer,
    Snip,
    Snips,
    Quantizer,
    Snipper,
)
