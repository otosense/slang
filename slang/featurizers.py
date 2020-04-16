from numpy import hanning, kaiser, abs, diff, array, std
from numpy.fft import rfft
from functools import wraps, partial
from slang.stypes import Chunk, Chunks, Featurizer


########################################################################################################################
# FFT

def identity_func(x):
    """The identify (a.k.a. transparent) function that returns it's input as is."""
    return x


def mk_window_func(window_func, *args, **kwargs):
    window_wf = window_func(*args, **kwargs)

    def wf_preproc(wf):
        return window_wf * wf

    return wf_preproc


mk_window_func.hanning = wraps(hanning)(partial(mk_window_func, hanning))
mk_window_func.kaiser = wraps(kaiser)(partial(mk_window_func, kaiser))


def mk_wf_to_spectr(preproc: callable = None,
                    fft_func: callable = rfft,
                    postproc: callable = abs):
    """Make a function that computes the spectrogram of a waveform
    By spectrum, we mean the output of the pipeline:
        `tile -> preproc -> fft -> postproc -> spectrum

    Because typically, we preprocess the input waveform (say, transform with a hanning function), and post process
    the fft (say take the norm of the complex vector).


    >>> import numpy as np
    >>> chk = np.ones(2048)  # the really interesting waveform we'll use as an example.
    >>> chk_size = len(chk)
    >>>
    >>> wf_to_spectr = mk_wf_to_spectr()  # default preproc is None (i.e. the waveform is not preprocessed)
    >>> s = wf_to_spectr(chk)
    >>> len(s)
    1025
    >>> assert s[1] == 0  # the second value is zero (with the hanning window, we wouldn't have that!)
    >>>
    >>> wf_to_spectr = mk_wf_to_spectr.w_hanning(chk_size)  # let's try the hanning window
    >>> s = wf_to_spectr(chk)
    >>> len(s)
    1025
    >>> assert s[1] > 0  # the second value is non-zero (without the hanning window, we wouldn't have that!)
    >>>
    >>> wf_to_spectr = mk_wf_to_spectr.w_kaiser(chk_size, beta=0.0)  # use kaiser preproc
    >>> s = wf_to_spectr(chk)
    >>> len(s)
    1025
    >>> assert s[1] == 0  # the second value is zero (because beta=0.0: with the hanning window, we wouldn't have that!)
    >>>
    """
    if callable(preproc):
        def wf_to_spectr(wf):
            return postproc(fft_func(preproc(wf)))
    else:
        def wf_to_spectr(wf):
            return postproc(fft_func(wf))

    return wf_to_spectr


def _mk_wf_to_spectr_w_hanning(window_size: int, fft_func: callable = rfft, postproc: callable = abs):
    """Make a wf_to_spectr function that uses a hanning window preproc.
    """
    preproc = mk_window_func.hanning(window_size)
    return mk_wf_to_spectr(preproc, fft_func, postproc)


def _mk_wf_to_spectr_w_kaiser(window_size: int, beta: float = 0.0,
                              fft_func: callable = rfft, postproc: callable = abs):
    """Make a wf_to_spectr function that uses a kaiser window preproc.

    The window_size should be set to the fixed tile (chunk) size you're using.
    The beta size is set to 0.0. To know what to set, look it up. We're using numpy's kaiser function for that.

    Here are some tips:

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hanning
    8.6   Similar to a Blackman
    ====  =======================

    """
    preproc = mk_window_func.kaiser(window_size, beta)
    return mk_wf_to_spectr(preproc, fft_func, postproc)


mk_wf_to_spectr.w_hanning = _mk_wf_to_spectr_w_hanning
mk_wf_to_spectr.w_kaiser = _mk_wf_to_spectr_w_kaiser

DFLT_WF_TO_SPECTR = mk_wf_to_spectr()

########################################################################################################################
# Spectral Projectors
from numpy import dot

mat_mult = dot


class NotFittedError(ValueError, AttributeError): ...


# Featurizer
class SpectralProjector:
    def __init__(self, scalings_, chk_to_spectr=DFLT_WF_TO_SPECTR):
        self.scalings_ = scalings_
        self.chk_to_spectr = chk_to_spectr

    @property
    def fv_length(self):
        return self.scalings_.shape[1]

    def spectr_mat(self, chks: Chunks):
        return array(list(self.chk_to_spectr(chk) for chk in chks))

    def transform(self, chks: Chunks):
        return mat_mult(self.spectr_mat(chks), self.scalings_)

    def fv_of_chk(self, chk):
        return mat_mult(self.chk_to_spectr(chk), self.scalings_)

    def __call__(self, chk: Chunk):
        return list(self.chk_to_spectr(chk))


########################################################################################################################
# Some silly featurizer to play with

from bisect import bisect_left


def zcr(chk):
    return sum(diff(array(chk) > 0).astype(int)) / (len(chk) - 1)


def rms_zcr(chk):
    return std(chk), zcr(chk)


_n_levels = 10
_zcr_level_dividers = [2 ** (-(x + 1)) for x in range(0, _n_levels - 1)][::-1]
_rms_level_dividers = [2048 * 2 ** (-(x + 1)) for x in range(0, _n_levels - 1)][::-1]


def _rms_zcr_to_levels(rms, zcr):
    return (bisect_left(_rms_level_dividers, rms),
            bisect_left(_zcr_level_dividers, zcr))


def rms_zcr_quantizer(fv):
    rms, zcr = fv
    rms_level, zero_crossing_level = _rms_zcr_to_levels(rms, zcr)
    return rms_level * _n_levels + zero_crossing_level


DFLT_FEATURIZER = rms_zcr
DFLT_QUANTIZER = rms_zcr_quantizer

import numpy as np
from typing import Union, Iterable


def mk_spectral_moment_featurizer(n_moments=100,
                                  preproc: callable = None,
                                  fft_func: callable = rfft,
                                  postproc: callable = abs):
    wf_to_spectr = mk_wf_to_spectr(preproc, fft_func, postproc)
    moments = np.arange(1, n_moments + 1)
    n_moments = len(moments)
    std_exponents = np.arange(1, n_moments + 1)

    if n_moments > 2:
        def moment_featurizer(wf):
            a = wf_to_spectr(wf)
            a_std = np.std(a)
            m = moment(a, moments) / (a_std ** std_exponents)
            m[0] = np.mean(a)
            m[1] = a_std
            return m
    elif n_moments == 2:
        def moment_featurizer(wf):
            a = wf_to_spectr(wf)
            return np.array([np.mean(a), np.std(a)])
    elif n_moments == 1:
        def moment_featurizer(wf):
            a = wf_to_spectr(wf)
            return np.array([np.mean(a)])
    else:
        raise ValueError(f"n_moments should be a positive integer. Instead, was {n_moments}")

    return moment_featurizer


def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis


def moment(a, moment: Union[int, Iterable] = 1, axis=0):
    r"""Calculate the nth moment about the mean for a sample.
    Taken from scipy.stats

    Examples
    --------
    >>> moment([1, 2, 3, 4, 5], moment=1)
    0.0
    >>> moment([1, 2, 3, 4, 5], moment=2)
    2.0
    """
    a, axis = _chk_asarray(a, axis)

    if a.size == 0:
        # empty array, return nan(s) with shape matching `moment`
        if np.isscalar(moment):
            return np.nan
        else:
            return np.full(np.asarray(moment).shape, np.nan, dtype=np.float64)

    # for array_like moment input, return a value for each.
    if not np.isscalar(moment):
        mmnt = [_moment(a, i, axis) for i in moment]
        return np.array(mmnt)
    else:
        return _moment(a, moment, axis)


def _moment(a, moment, axis):
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError("All moment parameters must be integers")

    if moment == 0:
        # When moment equals 0, the result is 1, by definition.
        shape = list(a.shape)
        del shape[axis]
        if shape:
            # return an actual array of the appropriate shape
            return np.ones(shape, dtype=float)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return 1.0

    elif moment == 1:
        # By definition the first moment about the mean is 0.
        shape = list(a.shape)
        del shape[axis]
        if shape:
            # return an actual array of the appropriate shape
            return np.zeros(shape, dtype=float)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return np.float64(0.0)
    else:
        # Exponentiation by squares: form exponent sequence
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        a_zero_mean = a - np.expand_dims(np.mean(a, axis), axis)
        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            s = a_zero_mean ** 2

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = s ** 2
            if n % 2:
                s *= a_zero_mean
        return np.mean(s, axis)
