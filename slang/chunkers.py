"""Chunker functions"""

from itertools import islice, chain
from typing import Callable
from functools import partial

inf = float('inf')

DFLT_CHK_SIZE = 2048

try:
    import numpy as np

    NUMPY_PRESENT = True
except (ImportError, ModuleNotFoundError):
    NUMPY_PRESENT = False


def fannout_chunker(
    objects,
    chunker,
    *,
    extract_chunkable: Callable,
    extract_info: Callable,
    include_chunk_key: bool = False
):
    """Chunk parts of an iterable of objects.

    Transform things like iterables of ``(info1, iterable, info2)`` to iterables
    ``(info2, iterable_chk_0), (info2, iterable_chk_1), ...``.

    :param objects: The iterable of objects that contain both the iterable to chunk and
    the extra info
    :param chunker: The chunker. A function taking the iterable to chunk and returns
    an iterable of chunks
    :param extract_chunkable: The function that extracts the iterable to be chunked
    :param extract_info: The function that extracts from the object the extra info to
    yield for each chunk
    :return: A generator of ``(info, chunk),...`` pairs or ``(chk_idx, (info, chunk))``
    pairs.

    Most of the time you'll be dealing with tuple or dicts as objects, so
    ``operator.itemgetter`` is your friend for creating the extractors.

    >>> from functools import partial
    >>> from operator import itemgetter
    >>> tuple_chunker = partial(
    ...     fannout_chunker,
    ...     chunker=partial(fixed_step_chunker, chk_size=2),
    ...     extract_chunkable=itemgetter(0),
    ...     extract_info=itemgetter(1))
    >>> objects = [([1,2,3,4], 'alice'), ([5,6,7], 'bob')]
    >>> list(tuple_chunker(objects))
    [('alice', [1, 2]), ('alice', [3, 4]), ('bob', [5, 6])]

    If you wanted to have get an index for your chunk as well, you can specify
    ``include_chunk_key=True``:

    >>> tuple_chunker_w_chk_index = partial(tuple_chunker, include_chunk_key=True)
    >>> list(tuple_chunker_w_chk_index(objects))
    [(0, ('alice', [1, 2])), (1, ('alice', [3, 4])), (0, ('bob', [5, 6]))]

    >>> dict_chunker = partial(
    ...     tuple_chunker,
    ...     extract_chunkable=itemgetter('wf'),
    ...     extract_info=itemgetter('tag'),
    ... )
    >>> objects = [
    ...     {'wf': [1,2,3,4], 'tag': 'alice', 'who_cares': 'about_this'},
    ...     {'wf': [5,6,7], 'tag': 'bob', 'or': 'this'}
    ... ]
    >>> list(dict_chunker(objects))
    [('alice', [1, 2]), ('alice', [3, 4]), ('bob', [5, 6])]

    The output format is fixed to ``(info, chunk),...`` or ``(chk_idx, (info, chunk))``.
    To get a different output, write egress function and apply it to output
    systematically. For example:

    >>> from i2 import Pipe
    >>> egress = lambda tup: {'tag': tup[0], 'chk': tup[1]}
    >>> new_dict_chunker = Pipe(dict_chunker, lambda x: map(egress, x))
    >>> assert list(new_dict_chunker(objects)) == [
    ...     {'tag': 'alice', 'chk': [1, 2]},
    ...     {'tag': 'alice', 'chk': [3, 4]},
    ...     {'tag': 'bob', 'chk': [5, 6]}
    ... ]
    """
    for obj in objects:
        iterable_to_chunk = extract_chunkable(obj)
        info = extract_info(obj)
        for chk_idx, chk in enumerate(chunker(iterable_to_chunk)):
            chk_and_info = (info, chk)
            if include_chunk_key:
                yield chk_idx, chk_and_info
            else:
                yield chk_and_info


def _matrix_of_chunks(a, chk_size):
    n = len(a)
    return np.reshape(a[: (n - (n % chk_size))], (-1, chk_size))


def _chunk_with_zip(a, chk_size):
    return zip(*([iter(a)] * chk_size))


def mk_chunker(chk_size=DFLT_CHK_SIZE, chk_step=None, *, use_numpy_reshape=None):
    """
    A fixed step chunker factory. Makes a function that takes an iterable and produces
    fixed-size fixed-step chunks of it's elements.

    This function-making function will try to choose an optimized chunker for you
    depending on the parameters and environment (if you have numpy and
    chk_size==chk_step, it's the fastest).

    Will also add attributes that exhibit the parameters of the chunker.

    Note though, that as a tradeoff, you may get numpy arrays, tuples, or lists as the
    type that is yield.

    :param chk_size: Size of chunk (default 2048)
    :param chk_step: Size of step (step of sliding window). If not specified, will be
    taken to be chk_size
    :param use_numpy_reshape: If None (default), will use numpy (reshape) if numpy is
    importable.
        If True, will try to use numpy.reshape systematically.
        If False, will not use numpy.reshape, even if numpy present.
    :return: A generator of chunks (numpy.arrays, tuples, or lists, depending on the context)

    >>> a = range(6)
    >>> chunker = mk_chunker(3)
    >>> list(chunker(a))
    [array([0, 1, 2]), array([3, 4, 5])]
    >>> list(mk_chunker(2)(a))
    [array([0, 1]), array([2, 3]), array([4, 5])]
    >>> list(tuple(x) for x in mk_chunker(1)(a))
    [(0,), (1,), (2,), (3,), (4,), (5,)]
    >>>
    >>> chunker = mk_chunker(4, 2)
    >>> list(chunker(a))
    [[0, 1, 2, 3], [2, 3, 4, 5]]

    See that ``chunker`` has parameters describing it:

    >>> chunker.chk_size
    4
    >>> chunker.chk_step
    2
    """

    chk_step = chk_step or chk_size  # default to chk_size == chk_step

    if chk_step == chk_size:
        if (use_numpy_reshape is None) or (use_numpy_reshape is True):
            if NUMPY_PRESENT:
                use_numpy_reshape = True
            else:
                if use_numpy_reshape is True:
                    raise ModuleNotFoundError(
                        "You don't have numpy installed, so you can't a reshape chunker"
                    )
                use_numpy_reshape = False

        if use_numpy_reshape:
            chunker = partial(_matrix_of_chunks, chk_size=chk_size)
        else:
            chunker = partial(_chunk_with_zip, chk_size=chk_size)

    else:
        chunker = partial(fixed_step_chunker, chk_size=chk_size, chk_step=chk_step)

    # Adding the chk_size and chk_step attributes to the chunker

    chunker.chk_size = chk_size
    chunker.chk_step = chk_step

    return chunker


DFLT_CHUNKER = mk_chunker(chk_size=DFLT_CHK_SIZE)


def simple_fixed_step_chunker(it, chk_size, chk_step=None):
    """
    Generates chunks of fixed size and step.
    Yields chunks as lists.
    """
    from itertools import islice

    if chk_step is None:
        chk_step = chk_size

    if chk_step == chk_size:
        yield from map(list, zip(*([iter(it)] * chk_step)))
    elif chk_step < chk_size:

        chk = list(islice(it, chk_size))

        while len(chk) == chk_size:
            yield chk
            chk = chk[chk_step:] + list(islice(it, chk_step))

    else:
        chk = list(islice(it, chk_size))
        gap = chk_step - chk_size

        while len(chk) == chk_size:
            yield chk
            chk = list(islice(it, gap, gap + chk_size))


def _validate_chk_size(chk_size):
    assert (
        isinstance(chk_size, int) and chk_size > 0
    ), 'chk_size should be a positive interger'


def _validate_chk_size_and_step(chk_size, chk_step):
    _validate_chk_size(chk_size)
    if chk_step is None:
        chk_step = chk_size
    assert (
        isinstance(chk_step, int) and chk_step > 0
    ), 'chk_step should be a positive integer'
    return chk_size, chk_step


def _validate_fixed_step_chunker_args(chk_size, chk_step, start_at, stop_at):
    chk_size, chk_step = _validate_chk_size_and_step(chk_size, chk_step)

    if start_at is None:
        start_at = 0
    if stop_at is not None:
        assert stop_at > start_at, 'stop_at should be larger than start_at'
        if stop_at is not inf:
            assert isinstance(stop_at, int), 'stop_at should be an integer'

    # checking a few things
    assert isinstance(start_at, int), 'start_at should be an integer'
    assert start_at >= 0, 'start_at should be a non negative integer'
    return chk_step, start_at


def rechunker(
    chks, chk_size, chk_step=None, start_at=None, stop_at=None, return_tail=False
):
    """Takes an iterable of chks and produces another iterable of chunks.
    The chunks generated by the input chks iterable is assumed to be gap-less and without overlap,
    but these do not need to be of fixed size.
    The output will be though.
    """
    yield from fixed_step_chunker(
        chain.from_iterable(chks), chk_size, chk_step, start_at, stop_at, return_tail
    )


def fixed_step_chunker(
    it, chk_size, chk_step=None, start_at=None, stop_at=None, return_tail=False
):
    """
      a function to get (an iterator of) segments (bt, tt) of chunks from an iterator (or list)
      of the for [it_1, it_2...], given a chk_size, chk_step, and a start_at and a stop_at.
      The start_at, stop_at act like slices indices for a list: start_at is included and stop_at is excluded

      :param it: iterator of elements of any type
      :param chk_size: length of the chunks
      :param chk_step: step between chunks
      :param start_at: index of the first term of the iterator at which we begin building the chunks (inclusive)
      :param stop_at: index of the last term from the iterator included in the chunks
      :param return_tail: if set to false, only the chunks with max element with index less than stop_at are yielded
      if set to true, any chunks with minimum index value no more than stop_at are returned but they contain
      term with index no more than stop_at
      :return: an iterator of the chunks

      1) If stop_at is not None and return_tail is False:
         will return all full chunks with maximum element index less than stop_at
         or until the iterator is exhausted. Only full chunks are returned here.

      2) If stop_at is not None and return_tail is True:
         will return all full chunks as above along with possibly cut off chunks
         containing one term whose index is stop_at-1 or one (last) term which is the
         last element of it

      3) If stop_at is None and return_tail is False:
         will return all full chunks with maximum element index less or equal to the last
         element of it

      4) If stop_at is None and return_tail is True:
         will return all full chunks with maximum element index less or equal to the last
         element of it plus cut off chunks whose maximum term index is the last term of it

        # testing chk_step < chk_size with return_tail=TRUE, stop and start_at PRESENT
        # and stop_at SMALLER than the largest index of it
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=2, stop_at=5, return_tail=True)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[3, 4, 5], [4, 5], [5]]

        # testing chk_step < chk_size with return_tail=FALSE, stop and start_at PRESENT
        # and stop_at SMALLER than the largest index of it
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=2, stop_at=5, return_tail=False)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[3, 4, 5]]

        # testing chk_step < chk_size with return_tail=TRUE, stop and start_at PRESENT
        # and stop_at LARGER than the largest index of it
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=1, stop_at=20, return_tail=True)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16], [15, 16], [16]]

        # testing chk_step < chk_size with return_tail=FALSE, stop and start_at PRESENT
        # and stop_at LARGER than the largest index of it
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=1, stop_at=20, return_tail=False)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16]]

        # testing chk_step = chk_size with return_tail=TRUE, stop and start_at PRESENT
        # and stop_at SMALLER than the largest index of it
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=3, start_at=1, stop_at=7, return_tail=True)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4], [5, 6, 7]]

        # testing chk_size > len(it) with return_tail=False, no stop_at or start_at
        >>> f = lambda it: fixed_step_chunker(it, chk_size=30, chk_step=3, start_at=None, stop_at=None, return_tail=False)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        []

        # testing chk_size > len(it) with return_tail=True, no stop_at or start_at
        >>> f = lambda it: fixed_step_chunker(it, chk_size=30, chk_step=3, start_at=None, stop_at=None, return_tail=True)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [10, 11, 12, 13, 14, 15, 16], [13, 14, 15, 16], [16]]

        # testing chk_step > chk_size with return_tail=TRUE, stop and start_at PRESENT
        # and stop_at SMALLER than the largest index of it
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=1, stop_at=7, return_tail=True)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4], [6, 7]]

        # testing chk_step > chk_size with return_tail=FALSE, stop and start_at PRESENT
        # and stop_at SMALLER than the largest index of it
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=1, stop_at=7, return_tail=False)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4]]

        # testing chk_step > chk_size with return_tail=FALSE, stop and start_at NOT PRESENT
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=False)
        >>> it = range(1, 17, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]]

        # testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=True)
        >>> it = range(1, 19, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18]]

        # testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
        # with negative values in the iterator
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=True)
        >>> it = range(-10, 19, 1)
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[-10, -9, -8], [-6, -5, -4], [-2, -1, 0], [2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16], [18]]

        # testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
        # with items of various types in the iterator
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=2, start_at=None, stop_at=None, return_tail=True)
        >>> it = ['a', 3, -10, 9.2, str, [1,2,3], set([10,20])]
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [['a', 3, -10], [-10, 9.2, <class 'str'>], [<class 'str'>, [1, 2, 3], {10, 20}], [{10, 20}]]
       """

    chk_step, start_at = _validate_fixed_step_chunker_args(
        chk_size, chk_step, start_at, stop_at
    )

    if chk_step == chk_size and not return_tail:
        yield from map(list, zip(*([iter(it)] * chk_step)))
    elif chk_step < chk_size:

        it = islice(it, start_at, stop_at)
        chk = list(islice(it, chk_size))

        while len(chk) == chk_size:
            yield chk
            chk = chk[chk_step:] + list(islice(it, chk_step))

    else:
        it = islice(it, start_at, stop_at)
        chk = list(islice(it, chk_size))
        gap = chk_step - chk_size

        while len(chk) == chk_size:
            yield chk
            chk = list(islice(it, gap, gap + chk_size))

    if return_tail:
        while len(chk) > 0:
            yield chk
            chk = chk[chk_step:]


#
# from collections import deque
# from itertools import islice, chain
#
#
# class nothing: ...
#
#
# strip_nothing_out = lambda item: tuple(filter(lambda x: x is not nothing, item))
#
#
# def new_fixed_step_chunker(it, chk_size, chk_step=None, start_at=None, stop_at=None, return_tail=False):
#     chk_step, start_at = _validate_fixed_step_chunker_args(chk_size, chk_step, start_at, stop_at)
#     if start_at is not None or stop_at is not None:
#         it = islice(it, start_at, stop_at)
#
#     iit = iter(it)
#     window = deque([], chk_size)
#     push_to_queue = window.extend
#
#     # and there after... push new step data to queue and yield queue contents
#     if not return_tail:
#         if chk_step == chk_size:
#             yield from zip(*([iit] * chk_step))
#         else:
#             chk = tuple(islice(iit, chk_size))
#             if len(chk) >= chk_size:
#                 yield chk
#                 push_to_queue(chk)
#                 for step_data in zip(*([iit] * chk_step)):
#                     push_to_queue(step_data)
#                     yield tuple(window)
#     else:
#         iiit = chain(iit, [nothing] * (chk_step - 1))
#         yield from map(strip_nothing_out, new_fixed_step_chunker(iiit, chk_size, chk_step, return_tail=False))
#         # step_nibs = zip(*([iiit] * chk_step))
#         # last_nib = next(step_nibs)
#         # for step_nib in step_nibs:
#         #     push_to_queue(last_nib)
#         #     yield tuple(window)
#         #     last_nib = step_nib
#         # push_to_queue(last_nib)
#         # last_part = list(window)
#         # yield strip_nothing_out(last_part)
