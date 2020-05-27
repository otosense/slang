import json
import os
from numpy import array, dot, cumsum, min, inf, concatenate, vstack, ndarray
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import NotFittedError
import inspect
from functools import partial
from omodel.ml.linalg_utils import residue
from omodel.core.spectro import Spectr

import numpy as np
from numpy import array, hanning, fft  # TODO: Get rid of this once we use C-based spectr

DFLT_WIN_FUNC = hanning
DFLT_CHK_SIZE = 2048
DFLT_MATRIX_MULTI = dot
DFLT_AMPLITUDE_FUNC = np.abs


def ascertain_array(iterable):
    """Getting an array from an iterable"""
    if not isinstance(iterable, ndarray):
        if not isinstance(iterable, list):
            iterable = list(iterable)
        iterable = array(iterable)
    return iterable


def mk_chk_fft(chk_size=None, window=DFLT_WIN_FUNC, amplitude_func=DFLT_AMPLITUDE_FUNC):
    """Make a chk_fft function that will compute the fft (with given amplitude and window).
    Note that this output chk_fft function will enforce a fixed chk_size (given explicitly, or through the
    size of the window (if window is given as an array)
    """
    if window is None:
        assert chk_size is not None, "chk_size must be a positive integer if window is a callable, or None"

        def chk_fft(chk):
            """Compute the power fft for a single chk"""
            assert len(chk) == chk_size, f"len(chk) was not the required chk_size={chk_size}"
            fft_amplitudes = amplitude_func(np.fft.rfft(chk))
            return fft_amplitudes
    else:
        if callable(window):
            assert chk_size is not None, "chk_size must be a positive integer if window is a callable, or None"
            window = window(chk_size)
        else:
            window = array(window)
            if chk_size is not None:
                assert len(window) == chk_size, f"chk_size ({chk_size}) and len(window) ({len(window)}) don't match"

        def chk_fft(chk):
            """Compute the power fft for a single chk"""
            fft_amplitudes = amplitude_func(np.fft.rfft(chk * window))
            return fft_amplitudes

    chk_fft.chk_size = chk_size
    return chk_fft


DFLT_CHK_FFT = mk_chk_fft(chk_size=DFLT_CHK_SIZE, window=DFLT_WIN_FUNC)


def instantiate_class_and_inject_attributes(cls, **kwargs):
    """
    instantiates a class with the given kwargs, picking those arguments that are in the signature of cls to
    use for the __init__, and adding attributes to the constructed object with the remaining.
    :param cls: class to insantiate
    :param kwargs: keyword args (some for the class __init__, and others to inject)
    :return: An (possibly enhanced) class instance
    >>> class C:
    ...     def __init__(self, a, b=3):
    ...         self.a = a
    ...         self.b = b
    ...
    >>> c = instantiate_class_and_inject_attributes(C, a=10, foo='bar')
    >>> c.__dict__
    {'a': 10, 'b': 3, 'foo': 'bar'}
    >>> c = instantiate_class_and_inject_attributes(C, a=10, foo='bar', bar='foo', b=1000)
    >>> c.__dict__
    {'a': 10, 'b': 1000, 'foo': 'bar', 'bar': 'foo'}
    >>> try:
    ...     c = instantiate_class_and_inject_attributes(C, foo='bar', bar='foo', b=1000)
    ... except TypeError:  # expected
    ...     pass
    """
    # cls_signature_args = inspect.signature(cls).parameters
    cls_signature_args = inspect.getfullargspec(cls.__init__).args[1:]

    cls_kwargs = dict()
    other_kwargs = dict()
    for k, v in kwargs.items():
        if k in cls_signature_args:
            cls_kwargs[k] = v
        else:
            other_kwargs[k] = v
    o = cls(**cls_kwargs)
    o.__dict__.update(other_kwargs)
    return o


class SpectralProjector:
    def __init__(self,
                 scalings_=None,
                 chk_fft=DFLT_CHK_FFT,
                 mat_mult=DFLT_MATRIX_MULTI):
        self.scalings_ = scalings_
        self.chk_fft = chk_fft
        self.mat_mult = mat_mult
        self.chk_size = getattr(chk_fft, 'chk_size', None)  # get chk_size from chk_fft if it has it

    def is_fitted(self):
        return hasattr(self, 'scalings_') and self.scalings_ is not None

    def assert_is_fitted(self):
        if not self.is_fitted():
            raise NotFittedError("{} was not fitted yet.".format(self.__class__.__name__))

    def spectras(self, chks):
        return array([self.chk_fft(chk) for chk in chks])

    def transform(self, chks):
        return self.mat_mult(self.spectras(chks), self.scalings_)

    def extend_scalings_(self, X_normal, X_outlier, n_extra_dims=1, update_scalings=False):
        """
        Add n_extra_dims vectors to the scaling matrix in order to better separate X_normal
        and X_outlier.
        :param X_normal: the feature vectors (=spectra for the intended case) of normal sounds
        :param X_outlier: the feature vectors of outlier sounds
        :param n_extra_dims: the number of extra dimensions to add to the projeccted fvs
        :param update_scalings: whether or not to update the original scalings with the new ones
        :return: the updated scalings matrix
        """
        X = concatenate([X_normal, X_outlier])
        y = array([0] * len(X_normal) + [1] * len(X_outlier))
        X_res = residue(self.scalings_.T, X)
        lda = LinearDiscriminantAnalysis(n_components=n_extra_dims)
        lda.fit(X_res, y)
        extra_proj_vec = lda.scalings_.flatten().reshape((n_extra_dims, lda.scalings_.shape[0]))
        new_scalings = vstack([self.scalings_.T, extra_proj_vec]).T
        if update_scalings:
            self.scalings_ = new_scalings
        return new_scalings

    def to_jdict(self):
        return {
            'scalings_': self.scalings_.tolist(),
        }

    @classmethod
    def from_jdict(cls, jdict):
        obj = instantiate_class_and_inject_attributes(cls, **jdict)
        if hasattr(obj, 'scalings_'):
            obj.scalings_ = array(obj.scalings_)
        return obj

    def __call__(self, chk):
        return self.transform([chk])


class TargettedVariancePCA:
    def __init__(self, target_variance=0.95, max_n_components=inf, min_n_components=1,
                 mat_mult=DFLT_MATRIX_MULTI):
        self.target_variance = target_variance
        self.max_n_components = max_n_components
        self.min_n_components = min_n_components
        self.mat_mult = mat_mult

    def fit(self, X, y=None):
        max_possible_comp = min(X.shape)
        self.max_n_components = int(min([self.max_n_components, max_possible_comp]))
        pca = PCA(n_components=self.max_n_components)
        pca.fit(X)
        exp_var_rat = pca.explained_variance_ratio_
        sum_exp_var_rat = cumsum(exp_var_rat)

        n_comp_to_retain = 0
        for n_comp_to_retain in range(self.min_n_components, self.max_n_components + 1):
            if sum_exp_var_rat[n_comp_to_retain - 1] >= self.target_variance:
                break

        pca.components_ = pca.components_[:n_comp_to_retain, :]

        # Note: pca not needed for the functioning of the class, but keeping around for debug reasons
        self._pca = pca

        self.scalings_ = pca.components_.T
        return self

    def transform(self, X):
        return self.mat_mult(X, self.scalings_)


class SpectralProjectorUnsupervisedFitter(SpectralProjector, TargettedVariancePCA):
    def __init__(self, target_variance=0.95, max_n_components=inf, min_n_components=1,
                 scalings_=None, chk_fft=DFLT_CHK_FFT, mat_mult=DFLT_MATRIX_MULTI):
        TargettedVariancePCA.__init__(self, target_variance, max_n_components, min_n_components, mat_mult)
        SpectralProjector.__init__(self, scalings_, chk_fft, mat_mult)

    def fit(self, chks, y=None):
        return super().fit(self.spectras(chks), y)


class MyLDA:
    def __init__(self, n_components=None, mat_mult=DFLT_MATRIX_MULTI):
        self.n_components = n_components
        self.mat_mult = mat_mult

    def fit(self, X, y):
        lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        lda.fit(X, ascertain_array(y))

        # Note: _lda not needed for the functioning of the class, but keeping around for debug reasons
        self._lda = lda

        self.scalings_ = lda.scalings_
        self.classes_ = lda.classes_
        return self

    def transform(self, X):
        return self.mat_mult(X, self.scalings_)


class SpectralProjectorSupervisedFitter(SpectralProjector, MyLDA):
    def __init__(self, n_components=None,
                 scalings_=None, chk_fft=DFLT_CHK_FFT, mat_mult=DFLT_MATRIX_MULTI):
        MyLDA.__init__(self, n_components, mat_mult)
        SpectralProjector.__init__(self, scalings_, chk_fft, mat_mult)

    def fit(self, chks, y):
        return super().fit(self.spectras(chks), y)
