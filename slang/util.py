from collections import deque, defaultdict
from itertools import islice
import numpy as np
from numpy import array, nan, arange, unique, ones
from numpy.random import choice
from slang.util_data import displayable_unichr

####### Seeing snips ###################################################################################################

unichr_code_of_snip = array(displayable_unichr
                            + list(unique(list(set(range(33, 20000)).difference(displayable_unichr)))))
snip_of_unichr_code = (nan * ones(unichr_code_of_snip.max() + 1)).astype(int)
snip_of_unichr_code[unichr_code_of_snip] = arange(len(unichr_code_of_snip))

# assert that snip_of_unichr_code is the inverse of unichr_code_of_snip
assert (all(snip_of_unichr_code[unichr_code_of_snip] == arange(len(unichr_code_of_snip))))


def snip_to_str(snip):
    return chr(unichr_code_of_snip[snip])


def snips_to_str(snips):
    return ''.join(map(chr, unichr_code_of_snip[list(snips)]))


def str_to_snips(snips_str):
    return array(snip_of_unichr_code[list(map(ord, snips_str))])


snips_of_str = str_to_snips  # alias for back-compatibility
str_of_snips = snips_to_str  # alias for back-compatibility

####### Misc ###########################################################################################################
ddir = lambda obj: [x for x in dir(obj) if not x.startswith('_')]


def balanced_sample_maker(key_to_tag, max_n_keys_per_tag=7, random=False):
    """making a sample of the data (when you want to just test quickly)

    >>> mk_sample = balanced_sample_maker(key_to_tag=lambda k: k.split('/')[0],
    ...                                   max_n_keys_per_tag=2,
    ...                                   random=False)
    >>> mk_sample(['good/1', 'bad/1', 'good/2', 'good/3', 'good/4', 'bad/2', 'good/5', 'bad/3'])
    ['good/1', 'good/2', 'bad/1', 'bad/2']
    """

    def mk_balanced_sample(keys):
        sample = defaultdict(list)
        for k in keys:
            sample[key_to_tag(k)].append(k)
        counts = {k: len(v) for k, v in sample.items()}
        min_count = min(max_n_keys_per_tag, min(counts.values()))
        sample_keys = list()
        if random is False:
            for k in sample:
                sample_keys.extend(sample[k][:min_count])
        else:
            for k in sample:
                sample_keys.extend(choice(sample[k], size=min_count, replace=False))
        return sample_keys

    return mk_balanced_sample


import operator


def _apply_op(op, d1, dflt_1, d2, dflt_2):
    if isinstance(d2, dict):
        out = dict()
        for k, v1 in d1.items():
            v2 = d2.get(k, dflt_2)
            out[k] = op(v1, v2)
        for k in d2:  # take care of the remainder (those keys in dict_2 that were not in dict_1)
            if k not in out:
                out[k] = op(dflt_1, d2[k])
    else:
        out = {k: op(v, d2) for k, v in d1.items()}
    return out


def _mk_op_method(op, dflt_1, dflt_2, for_reflexive_op=False):
    if not for_reflexive_op:
        def op_method(self, d):
            return self.__class__(_apply_op(op, self, dflt_1, d, dflt_2))
    else:
        def op_method(self, d):
            return self.__class__(_apply_op(op, d, dflt_1, self, dflt_2))

    return op_method


def _mk_unary_op_method(op):
    def op_method(self):
        return self.__class__({k: op(v) for k, v in self.items()})

    return op_method


# Notes: Not sure if mod has an identity.
# __and__ would have one theoretically (infinity 1s). didn't think about xor.
# __matmul__ has an identity DEPENDING ON square and if so, PER dimensions of matrix
# TODO: not sure of the appropriate defaults for __pow__.
_ops_and_identity = [
    ({'__add__', '__sub__', '__lshift__', '__rshift__', '__or__'}, 0),
    ({'__mul__', '__truediv__', '__floordiv__', '__pow__'}, 1),
    ({'__mod__', '__and__', '__xor__', '__matmul__'}, None)
]

_unary_ops = {'__pos__', '__neg__', '__abs__', '__invert__'}


class ArithmeDict(dict):
    """A dict, with arithmetic.
    A unary operator is just applied to all values.
    When a dict operates with a number, the operation is applied to each value of the dict.
    When a dict operates with another dict, the keys are aligned and the operation applied to the aligned values.

    The class is meant to be used in situations where pandas.Series would be used to operate with (sparse) vectors
    such as word counts, etc.

    Performance:

    In a nutshell, if you use pandas already in your app, then use pandas.Series instead.
    But, if you want weight packages (pandas isn't light), or have small dicts you want to operate on, use ArithmeDict.

    Note that both construction and operation are faster on ArithmeDict, for smaller sets.

    ```
    import pandas as pd

    t = ArithmeDict(a=1, b=2)
    tt = ArithmeDict(b=3, c=4)
    %timeit t + tt
    # 1.41 µs ± 41.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    ### versus ###

    t = pd.Series(dict(a=1, b=2))
    tt = pd.Series(dict(b=3, c=4))
    %timeit t + tt  # and not even what we want (see later)
    # 405 µs ± 7.65 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    % timeit pd.Series.add(t, tt, fill_value=0).to_dict()
    # 410 µs ± 11.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    ### but ###
    t = ArithmeDict({i: i for i in range(10000)})
    tt = ArithmeDict({i: i for i in range(5000, 15000)})
    %timeit t + tt
    # 3.22 ms ± 98.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    ### not so far from ###
    t = pd.Series({i: i for i in range(10000)})
    tt = pd.Series({i: i for i in range(5000, 15000)})
    %timeit pd.Series.add(t, tt, fill_value=0).to_dict()
    3.71 ms ± 100 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # and actually much slower than:
    %timeit pd.Series.add(t, tt, fill_value=0)
    575 µs ± 17.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    ```

    On the other hand, memory usage is inconclusive, because I don't know how to actually make the comparison.
    ```
    import pickle, sys, pandas

    t = ArithmeDict({i: i for i in range(10000)})
    sys.getsizeof(t), len(pickle.dumps(t))
    # (295032, 59539)

    t = pandas.Series({i: i for i in range(10000)})
    sys.getsizeof(t), len(pickle.dumps(t))
    # (160032, 240666)
    ```


    Notes for enhancement:

    When dict operates with/on a dict, and therefore we need to align keys,
    there are different merge and reduce options that may or may not make sense according to the value type and context.
    For example, should we really keep all keys and use operand defaults to get their values, or just drop
    those fields all together?
    Also, if we choose to keep all keys, what should the operand default be.
    Sometimes it might depend on the other operand (example matmul), or need to be created (example __concat__,
    since don't want the mutable list as a default), etc.

    >>> d1 = ArithmeDict(a=1, b=2)
    >>> d2 = ArithmeDict(b=3, c=4)
    >>>
    >>> # These are still dicts
    >>> isinstance(d1, dict)
    True
    >>> # and display as such
    >>> d1
    {'a': 1, 'b': 2}
    >>> d2
    {'b': 3, 'c': 4}
    >>>
    >>> # Unary operators (just applied to all values)
    >>> assert -d1 == {'a': -1, 'b': -2}
    >>> assert abs(-d1) == d1  # ... and in case that doesn't look impressive enough..
    >>> assert abs(ArithmeDict(a=-1, b=2, c=-3)) == {'a': 1, 'b': 2, 'c': 3}
    >>>
    >>> # An operation with a number is transferred to the values of the dict (applied to each).
    >>> assert d1 + 10 == {'a': 11, 'b': 12}
    >>> assert d1 - 10 == {'a': -9, 'b': -8}
    >>> assert d1 * 10 == {'a': 10, 'b': 20}
    >>> assert d1 / 10 == {'a': 0.1, 'b': 0.2}
    >>> assert d1 // 2 == {'a': 0, 'b': 1}
    >>> assert d1 ** 2 == {'a': 1, 'b': 4}
    >>> assert d2 % 2 == {'b': 1, 'c': 0}
    >>> assert d2 % 3 == {'b': 0, 'c': 1}
    >>> assert d2 >> 1 == {'b': 1, 'c': 2}  # shift all bits by one bit to the right
    >>> assert d2 << 1 == {'b': 6, 'c': 8}  # shift all bits by one bit to the left
    >>>
    >>> # An operation with another dict will align the keys and apply the operation to the aligned values.
    >>> assert d1 + d2 == {'a': 1, 'b': 5, 'c': 4}
    >>> assert d1 - d2 == {'a': 1, 'b': -1, 'c': -4}
    >>> assert d1 * d2 == {'a': 1, 'b': 6, 'c': 4}
    >>> assert d1 / d2 == {'a': 1, 'b': 0.6666666666666666, 'c': 0.25}
    >>> assert d2 // d1 == {'b': 1, 'c': 4, 'a': 1}
    >>> assert d1 ** d2 == {'a': 1, 'b': 8, 'c': 1}
    >>> assert ArithmeDict(a=10, b=10) % dict(a=3, b=4) == {'a': 1, 'b': 2}
    >>> assert d1 << d2 == {'a': 1, 'b': 16, 'c': 0}  # shifting bits
    >>> assert d1 + {'b': 3, 'c': 4} == {'a': 1, 'b': 5, 'c': 4}  # works when the right side is a normal dict
    >>> assert d1 + ArithmeDict() == d1
    >>> assert ArithmeDict() - d1 == -d1
    """

    for op in _unary_ops:
        locals()[op] = _mk_unary_op_method(getattr(operator, op))

    for ops, identity_val in _ops_and_identity:
        for op in ops:
            op_func = getattr(operator, op)
            locals()[op] = _mk_op_method(op_func, dflt_1=identity_val, dflt_2=identity_val, for_reflexive_op=False)


def running_mean_gen(it, chk_size=2, chk_step=1):  # TODO: A version of this with chk_step as well
    """
    Running mean (moving average) on iterator.
    Note: When input it is list-like, ut.stats.smooth.sliders version of running_mean is 4 times more efficient with
    big (but not too big, because happens in RAM) inputs.
    :param it: iterable
    :param chk_size: width of the window to take means from
    :return:

    >>> list(running_mean([1, 3, 5, 7, 9], 2))
    [2.0, 4.0, 6.0, 8.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 2, chk_step=2))
    [2.0, 6.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 2, chk_step=3))
    [2.0, 8.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 3))
    [3.0, 5.0, 7.0]
    >>> list(running_mean([1, -1, 1, -1], 2))
    [0.0, 0.0, 0.0]
    >>> list(running_mean([-1, -2, -3, -4], 3))
    [-2.0, -3.0]
    """
    if chk_step is None:
        chk_step = chk_size
    if chk_step > 1:
        # TODO: perhaps there's a more efficient way. A way that would sum the values of every step and add them in bulk
        yield from islice(running_mean(it, chk_size), None, None, chk_step)
    else:
        it = iter(it)
        if chk_size > 1:

            c = 0
            fifo = deque([], maxlen=chk_size)
            for i, x in enumerate(it, 1):
                fifo.append(x)
                c += x
                if i >= chk_size:
                    break

            yield c / chk_size

            for x in it:
                c += x - fifo[0]  # NOTE: seems faster than fifo.popleft
                fifo.append(x)
                yield c / chk_size
        else:
            for x in it:
                yield x


def running_mean(arr, chk_size):
    c = np.cumsum(np.insert(arr, 0, [0]))
    return (c[chk_size:] - c[:-chk_size]) / chk_size
