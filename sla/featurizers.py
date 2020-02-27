from numpy import hanning, kaiser, abs, diff, array, std
from numpy.fft import rfft
from functools import wraps, partial


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
# Making a default nipper

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


def rms_zcr_nipper(fv):
    rms, zcr = fv
    rms_level, zero_crossing_level = _rms_zcr_to_levels(rms, zcr)
    return rms_level * _n_levels + zero_crossing_level


#
# def _zcr_level(chk):
#     return bisect_left(_zcr_level_dividers, _zero_crossing_ratio(chk))
#
#
# def _rms_level(chk):
#     return bisect_left(_rms_level_dividers, std(chk))
#
#
#
# def _rms_and_crossing_levels(chk):
#     rms_level = int(min(_n_levels - 1, _n_levels * std(chk) / 500))
#     return rms_level, _zcr_level(chk)
#
#
# def _rms_and_crossing_levels(chk):
#     rms_level = int(min(_n_levels - 1, _n_levels * std(chk) / 500))
#     return rms_level, _zcr_level(chk)
#
#
# def rms_zcr_to_levels(rms, zcr):
#     return bisect_left(_rms_level_dividers, rms), bisect_left(_zcr_level_dividers, zcr)
#
# def rms_and_crossing_nipper(chk):
#     rms_level, zero_crossing_level = _rms_and_crossing_levels(chk)
#     return rms_level * _n_levels + zero_crossing_level


DFLT_FEATURIZER = rms_zcr
DFLT_NIPPER = rms_zcr_nipper
