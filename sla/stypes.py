import attr
from typing import Callable, Union, Iterable, Any, List, Tuple
import typing as ty
from numpy import ndarray, int16, int32, float32, float64
# Note I'm using Tuple to denote fixed sized sequences and List to denote a sliceable unbounded sized iterator and
# tuple as a fixed


from typing import TypeVar, Optional, Iterable, Sequence


def MyType(name: str, constraint, *more_constraints, doc: Optional[str] = None, aka: Optional[Iterable] = None,
           bound=None, covariant=False, contravariant=False,
           assign_to_globals=False):
    """
    Make a new type with (optional) doc and (optional) aka, set of var names it often appears as

    Args:
        name: Name to give the variable
        constraints: types (see typing.TypeVar)
        doc: Optional string to put in __doc__ attribute
        aka: Optional set (or any iterable) to put in _aka attribute,
            meant to list names the variables of this type often appear as.

    Returns: None

    >>> from typing import Any, List
    >>> T = MyType('T', int)
    >>> type(T)
    <class 'function'>
    >>> Key = MyType('Key', Any, aka=['key', 'k'])
    >>> Key._aka
    {'key', 'k'}
    >>> Val = MyType('Val', int, float, List[Union[int, float]], doc="A number or list of numbers.")
    >>> Val.__doc__
    'A number or list of numbers.'
    """
    if len(more_constraints) == 0:
        new_tp = TypeVar(name, bound=constraint, covariant=covariant, contravariant=contravariant)
    else:
        new_tp = TypeVar(name, constraint, *more_constraints, covariant=covariant, contravariant=contravariant)
    if doc is not None:
        setattr(new_tp, '__doc__', doc)
    if aka is not None:
        setattr(new_tp, '_aka', set(aka))
    if assign_to_globals:
        globals()[name] = new_tp  # not sure how kosher this is... Should only use at top level of module, for sure!
    return new_tp


# This would have been convenient, but pycharm doesn't see the globals created by NT!
# from functools import partial
# NT = partial(new_type, assign_to_globals=True)


FixedSizeSeq = MyType('FixedSizeSeq', Tuple, List, ndarray)

VarSizeSeq = MyType('VarSizeSeq', List, ndarray)

Key = MyType('Key', Any, doc="Any object used to reference another", aka={'key', 'k'})

# Waveform = Iterable[Union[float, int]]
Sample = MyType('Sample', float, int, int16, int32, float32, float64,
                doc="The numerical value of a digital signal sample")

# Waveform = MyType('Waveform', VarSizeSeq[Sample])
Waveform = MyType('Waveform', Sequence[Sample])
Waveforms = Iterable[Waveform]

# WfGen = MyType('WfGen', Iterable[Waveform], doc='A iterable of Waveforms')
KeyWfGen = MyType('KeyWfGen', Iterable[Tuple[Key, Waveform]], doc='A iterable of (Key, Waveform) pairs')

# Chunk = MyType('Chunk', FixedSizeSeq[Sample])
Chunk = MyType('Chunk', Sequence[Sample])
Chunks = Iterable[Chunk]
Chunker = Callable[[Waveform], Iterable[Chunk]]

Feature = MyType('Feature', float, int, int16, int32, float32, float64)
# FV = FixedSizeSeq[Feature]
FV = Sequence[Feature]
Featurizer = Callable[[Any], FV]
ChkFeaturizer = Callable[[Chunk], FV]
# ChkFeaturizer

Snip = int  # but with an upper limit... From an "alphabet"
Snips = Iterable[Snip]
Quantizer = Callable[[Any], Snip]

Snipper = Callable[[Waveform], Snips]
