"""Spectral projectors"""

from functools import wraps, partial
import inspect
from dataclasses import dataclass, field
from typing import Union, Optional, Callable
import warnings
from math import sqrt

from numpy import cumsum, min, inf, ndarray, floor
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from numpy import (
    array,
    hanning,
    fft,
)  # TODO: Get rid of this once we use C-based spectr

DFLT_WIN_FUNC = hanning
DFLT_CHK_SIZE = 2048
DFLT_INPUT_SIZE = int(1 + DFLT_CHK_SIZE / 2)  # 1025 for chk sie of 2048
DFLT_N_FREQ = DFLT_INPUT_SIZE  # backcompat alias
DFLT_AMPLITUDE_FUNC = np.abs
DFLT_FREQ_RANGE = (0, DFLT_N_FREQ)


def make_band_matrix_row(list_entries, row_len):
    """
    Makes a row for the spectral bucket matrix. The row is zero everywhere except on the entries of index in
    list_entries where it is 1 / len(list_entries)

    :param list_entries: the indices of non zero entries of the row
    :param row_len: the length of the row
    :return: an array of length row_len as described above

    >>> make_band_matrix_row([3], 5)
    array([0., 0., 0., 1., 0.])
    >>> make_band_matrix_row([0, 3], 5)
    array([0.5, 0. , 0. , 0.5, 0. ])

    """

    n_non_zero = len(list_entries)
    row = np.array(
        [0 if i not in list_entries else 1 / n_non_zero for i in range(row_len)]
    )
    return row


def make_band_matrix(buckets, n_freq=DFLT_INPUT_SIZE):
    """
    Given a list of n list of indices, make a matrix of size n by n_freq where the entries of row k are 0
    everywhere except at the index in the k list of buckets, where the entries are the inverse of the length
    of that bucket

    :param buckets: a list of list containing the indices of non zero entries of the corresponding row
    :param n_freq: the number of column of the matrix
    :return: a len(buckets) by n_freq matrix

    >>> buckets = make_buckets(n_buckets=15, freqs_weighting=lambda x: np.log(x + 0.001), freq_range=(200, 1000))
    >>> M = make_band_matrix(buckets, n_freq=1025)
    >>> print(M.shape)
    (15, 1025)

    >>> # the matrix sends each of the 200 first unitary vector to the zero vector (not below is not a mathematical proof of that, but solid clue)
    >>> vec = [1] * 200 + [0] * (1025 - 200)
    >>> print(np.dot(M, vec))
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    >>> # after that, we get non zero output (in general at least)
    >>> vec = [1] * 200 + [1] + [0] * (1025 - 201)
    >>> print(np.dot(M, vec))
    [0.01612903 0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.        ]

    """

    n_bands = len(buckets)
    bucket_matrix = np.zeros((n_bands, n_freq))
    for row_idx, bucket in enumerate(buckets):
        row = make_band_matrix_row(list_entries=bucket, row_len=n_freq)
        bucket_matrix[row_idx] = row
    return bucket_matrix


def decreasing_integer_geometric_sequence(
    start: int = DFLT_INPUT_SIZE, scale_factor=0.5
) -> list:
    """Generate decreasing positive integers in by multiplying numbers by a constant (between 0 and 1) repeatedly.
    Numbers of ther sequence will all be integers and not repeat (so often not a true geometri sequence).
    All sequences will end with 1.
    An error will be raised if scale_factor is not between 0 and 1 (exclusive).

    >>> decreasing_integer_geometric_sequence(128)
    [128, 64, 32, 16, 8, 4, 2, 1]
    >>> decreasing_integer_geometric_sequence(10, 0.3)
    [10, 3, 1]
    >>> decreasing_integer_geometric_sequence(10, 0.7)
    [10, 7, 5, 3, 2, 1]

    And to see that we indeed don't get duplicates of the true geometric sequence.

    >>> decreasing_integer_geometric_sequence(10, 0.9999)
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    """
    assert (
        0 < scale_factor < 1
    ), 'This geometric_sequence is meant for decreasing sequences only'

    def gen():
        cursor = start
        yield round(cursor)
        while cursor > 1:
            cursor *= scale_factor
            yield round(cursor)

    return list(dict.fromkeys(gen()))


# TODO: Continue algorithm further, filling more coarse coverage with static greedy
#  rule, thus making n_buckets unbound
def logarithmic_bands_matrix(
    n_buckets: Optional[int] = None,
    n_freqs: int = DFLT_INPUT_SIZE,
    factor: Optional[float] = None,
) -> np.ndarray:
    """Makes a spectral projection matrix that puts more importance on low frequencies than high ones.
    Importance both in weight and in precision.
    By a factor of 2 by default, but can be any amount.
    Note that the factor here is the inverse of the geometric factor of decreasing_integer_geometric_sequence.

    Note that the sum of the frequencies of the bands is constant from band to band. Not sure if that's the best choice,
    or if equal weight through out is the best choice.
    We can discuss what both mean at a later date.

    Important here is to understand my intent, to see how well I achieve it.
    - The features "build on each other". That is, if you ask (within the same
    `(n_freqs, factor)` set) for 7 features in your fv, the first 5 will be the same
    as if you asked for only 5. This makes it easier to compare
    fvs between projects, and possibly even "add to existing features".
    - Note  that the first feature is always total energy.
    - Lower frequencies are given less importance -- both in precision and in weight.
    - The linear algebraists that are listening will note what the vector space
    generated actually is, compared to that created by disjoint bands.

    >>> print(*logarithmic_bands_matrix(n_freqs=8).tolist(), sep='\\n')
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> print(*logarithmic_bands_matrix(n_freqs=8, factor=1.5).tolist(), sep='\\n')
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
    [0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    """
    if n_buckets is None and factor is None:
        factor = 2
    bucket_upper_bounds = _compute_bucket_upper_bounds(n_buckets, n_freqs, factor)

    buckets = list(range(0, i) for i in bucket_upper_bounds)
    m = make_band_matrix(buckets, n_freq=n_freqs)
    if n_buckets is not None:
        assert n_buckets <= len(m), (
            f'you asked for {n_buckets}, but the matrix has only {len(m)} rows. '
            f'Consider decreasing the factor to get more buckets'
        )
    else:
        n_buckets = len(m)
    return m[:n_buckets]


def _compute_bucket_upper_bounds(n_buckets, n_freqs, factor, max_tries=20):
    """Computes bucket upper bounds, trying to compensate for float numerical errors.

    If factor is None, the formula to get the smallest factor that is needded to get
    the given ``n_buckets`` is ``factor = n_freqs ** (1 / (n_buckets - 1))``.
    Unfortunately, this theoretical truth doesn't reflect in physical computation
    due to float precision errors. Our hack here is to ask for the factor for a higher
    number of buckets until we get a factor that is "sufficient".
    """
    if factor is None:
        # TODO: This formula is not bullet proof. We tried to compensate for numerical
        #  errors for reasonably small n_buckets and n_freqs
        for excess in range(max_tries):
            factor = n_freqs ** (1 / (n_buckets + excess - 1))
            bucket_upper_bounds = decreasing_integer_geometric_sequence(
                n_freqs, 1 / factor
            )
            if len(bucket_upper_bounds) > n_buckets:
                return bucket_upper_bounds
        if excess == max_tries - 1:
            raise ValueError(f'Maximum number of tries exceeded')

    else:
        return decreasing_integer_geometric_sequence(n_freqs, 1 / factor)


# A few default spectral projection matrices
geo_mat_11 = logarithmic_bands_matrix(11, n_freqs=1025, factor=2)
geo_mat_17 = logarithmic_bands_matrix(17, n_freqs=1025, factor=1.5)
geo_mat_34 = logarithmic_bands_matrix(34, n_freqs=1025, factor=1.2)

DFLT_SCALINGS = geo_mat_11


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


def ascertain_array(iterable):
    """Getting an array from an iterable"""
    if iterable is not None and not isinstance(iterable, ndarray):
        if not isinstance(iterable, list):
            iterable = list(iterable)
        iterable = array(iterable)
    return iterable


def chk_to_spectrum(
    chk, chk_size, window=DFLT_WIN_FUNC, amplitude_func=DFLT_AMPLITUDE_FUNC
):
    assert len(chk) == chk_size, (
        f'This function was made for chk_size={chk_size}. '
        f'You fed a chk of size len(chk)={len(chk)} instead'
    )
    fft_amplitudes = amplitude_func(np.fft.rfft(chk * window))
    return fft_amplitudes


def named_partial(name, func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    partial_func.name = name
    return partial_func


def mk_chk_fft(chk_size=None, window=DFLT_WIN_FUNC, amplitude_func=DFLT_AMPLITUDE_FUNC):
    """Make a chk_fft function that will compute the fft (with given amplitude and window).
    Note that this output chk_fft function will enforce a fixed chk_size (given explicitly, or through the
    size of the window (if window is given as an array)

    >>> chk_size = 4 * 5
    >>> f = mk_chk_fft(chk_size)
    >>> chk = 4 * list(range(chk_size // 4))
    >>> f(chk)
    array([19.        , 10.214421  ,  0.40774689,  4.34150779,  8.09801978,
            4.30684381,  0.26711259,  2.61514496,  5.04588504,  2.6255033 ,
            0.21962565])
    >>> # verifying that it's pickable
    >>> import pickle
    >>> import numpy as np
    >>> ff = pickle.loads(pickle.dumps(f))
    >>> assert np.allclose(ff(chk), ff(chk))
    """
    if callable(window):
        assert (
            chk_size is not None
        ), 'chk_size must be a positive integer if window is a callable, or None'
        window = window(chk_size)
    elif window is None:
        window = 1
    else:
        window = np.array(window)
        if chk_size is not None:
            assert (
                len(window) == chk_size
            ), f"chk_size ({chk_size}) and len(window) ({len(window)}) don't match"

    chk_spectrum = named_partial(
        'chk_to_spectrum',
        chk_to_spectrum,
        chk_size=chk_size,
        window=window,
        amplitude_func=amplitude_func,
    )
    chk_spectrum.chk_size = chk_size
    return chk_spectrum


DFLT_CHK_FFT = mk_chk_fft(chk_size=DFLT_CHK_SIZE, window=DFLT_WIN_FUNC)

matrix_mult = np.dot
DFLT_MATRIX_MULTI = matrix_mult


# TODO: Does the doc match what the function does?
# TODO: Better name for this, to distinguish between expressing projection in projected sub-space or original space
#   The following expresses it in the original space
def projection(basis, vectors, mat_mult=DFLT_MATRIX_MULTI):
    """
    The vectors live in a k dimensional space S and the columns of the basis are vectors of the same
    space spanning a subspace of S. Gives a representation of the projection of vector into the space
    spanned by basis in term of the basis.

    :param basis: an n-by-k array, a matrix whose vertical columns are the vectors of the basis
    :param vectors: an m-by-k array, a vector to be represented in the basis
    :param mat_mult: the function to multiply matrices
    :return: an m-by-k array
    """
    return mat_mult(mat_mult(vectors, basis), basis.T)


def reducing_proj(basis, vectors, mat_mult=DFLT_MATRIX_MULTI):
    """What we actually use to get fvs from spectras"""
    return mat_mult(vectors, basis.T)


def _assert_spectrum_and_chk_size_match(spectrum_size: int, chk_size: int):
    assert spectrum_size == int(1 + chk_size / 2), (
        f'expected spectrum_size == int(1 + chk_size / 2): '
        f'{spectrum_size=} and {chk_size=}'
    )


def _chk_size_to_spectrum_size(chk_size: int) -> int:
    return int(1 + chk_size / 2)


def _spectrum_size_to_chk_size(spectrum_size: int) -> int:
    if spectrum_size == 1:
        return 1
    return 2 * spectrum_size - 2


def _get_spectrum_size_and_validate_against_chk_size(spectrum_size, chk_size):
    assert (
        chk_size is not None or spectrum_size is not None
    ), 'either chk_size or spectrum size must be given'
    if spectrum_size is None:
        if chk_size is not None:
            spectrum_size = int(1 + chk_size / 2)
        else:
            _assert_spectrum_and_chk_size_match(spectrum_size, chk_size)
    return spectrum_size


@dataclass
class Projector:
    scalings_: np.ndarray = DFLT_SCALINGS
    mat_mult: Callable = DFLT_MATRIX_MULTI

    def transform(self, X):
        # return projection(self.scalings_, ascertain_array(X))
        return reducing_proj(self.scalings_, ascertain_array(X), mat_mult=self.mat_mult)

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

    def is_fitted(self):
        return hasattr(self, 'scalings_') and self.scalings_ is not None

    def assert_is_fitted(self):
        if not self.is_fitted():
            raise NotFittedError(
                '{} was not fitted yet.'.format(self.__class__.__name__)
            )

    def __call__(self, vector):
        return self.transform([vector])[0]


@dataclass
class SpectralProjector(Projector):
    chk_fft: Callable = DFLT_CHK_FFT

    def __post_init__(self):
        if not hasattr(self, 'chk_size'):
            self.chk_size = getattr(
                self.chk_fft, 'chk_size', None
            )  # get chk_size from chk_fft if it has it

    # TODO: spectrum_size is elsewhere n_freqs (should we be consistent?)
    @classmethod
    def for_sizes(
        cls,
        chk_size: Optional[int] = None,
        n_features: Optional[int] = None,
        *,
        spectrum_size: Optional[int] = None,
        log_factor: Optional[float] = None,
    ):
        """Makes a projector for given (chunk or spectrum) and feature space size.
        This is a convenience to get a projector given only the input
        (chunk or spectrum) size, and optionally output (feature space) size.
        That is, we only need minimal information about the data here, not to fit
        on the data itself.

        >>> featurizer = SpectralProjector.for_sizes(chk_size=20)
        >>> featurizer.scalings_.shape
        (4, 11)
        >>> SpectralProjector.for_sizes(chk_size=21).scalings_.shape
        (4, 11)
        >>> SpectralProjector.for_sizes(chk_size=21, n_features=2).scalings_.shape
        (2, 11)
        >>> SpectralProjector.for_sizes(spectrum_size=21).scalings_.shape
        (5, 21)

        >>> featurizer = SpectralProjector.for_sizes(chk_size=4)
        >>> chk = [1, 2, 3, 4]
        >>> fv = featurizer(chk).round(decimals=3)
        >>> fv
        array([2.401, 3.227])

        >>> spectrum = featurizer.spectras([chk])[0]
        >>> spectrum.round(decimals=2)
        array([3.75, 2.7 , 0.75])
        >>> featurizer.scalings_.round(2)
        array([[0.33, 0.33, 0.33],
               [0.5 , 0.5 , 0.  ]])
        >>> fv == featurizer.mat_mult(featurizer.scalings_, spectrum).round(decimals=3)
        array([ True,  True])

        """
        spectrum_size = _get_spectrum_size_and_validate_against_chk_size(
            spectrum_size=spectrum_size, chk_size=chk_size
        )
        # TODO: Research some scientifically based default n_features based on spectrum
        n_features = n_features or min((50, int(1 + sqrt(spectrum_size))))
        scalings_ = logarithmic_bands_matrix(
            n_buckets=n_features, n_freqs=spectrum_size, factor=log_factor
        )
        instance = cls(scalings_)
        if chk_size is not None:
            instance.chk_size = chk_size
        instance.spectrum_size = spectrum_size
        instance.chk_fft = mk_chk_fft(chk_size=instance.chk_size, window=DFLT_WIN_FUNC)
        return instance

    def spectras(self, chks):
        return array([self.chk_fft(chk) for chk in chks])

    def transform(self, chks):
        return super().transform(self.spectras(chks))

    def __call__(self, chk):
        return self.transform([chk])[0]

    mk_chk_fft = staticmethod(
        mk_chk_fft
    )  # to have it available to make a chk_fft in __init__


# @classmethod
# def for_chk_size(cls, scalings_=None,
#                  chk_size=None, window=DFLT_WIN_FUNC, amplitude_func=DFLT_AMPLITUDE_FUNC,
#                  mat_mult=DFLT_MATRIX_MULTI):
#     chk_fft = mk_chk_fft(chk_size, window, amplitude_func)
#     return cls(scalings_, chk_fft, mat_mult)


# def handle_iterables(learner_cls):
#     @wraps(learner_cls.fit)
#     def fit(self, X, y=None, *args, **kwargs):
#         # TODO use signature bind to extract the X and y from args, and kwargs (and get rid of X and y explicit arg)
#         try:
#             self.fit(X, y, *args, **kwargs)
#         except Exception:  # TODO: Be less broad
#             # Use chunking to fit partial in batches (probably more efficient)
#             if hasattr(self, 'fit_partial'):
#                 for x_item, y_item in zip(X, y):
#                     self.fit_partial(x_item, y_item)
#             else:
#                 X = ascertain_array(X)
#                 y = ascertain_array(y)
#                 self.fit(X, y, *args, **kwargs)
#         return self
#     learner_cls.fit = fit


def fit_handling_iterables(learner, X, y=None, *args, **kwargs):
    try:
        learner.fit(X, y, *args, **kwargs)
    except Exception:  # TODO: Be less broad
        # Use chunking to fit partial in batches (probably more efficient)
        if hasattr(learner, 'fit_partial'):
            for x_item, y_item in zip(X, y):
                learner.fit_partial(x_item, y_item)
        else:
            X = ascertain_array(X)
            y = ascertain_array(y)
            learner.fit(X, y, *args, **kwargs)
    return learner


class SpectralProjectorLearner(SpectralProjector, BaseEstimator, TransformerMixin):
    def __init__(self, learner, chk_fft=DFLT_CHK_FFT, mat_mult=DFLT_MATRIX_MULTI):
        super().__init__(scalings_=None, chk_fft=chk_fft, mat_mult=mat_mult)
        self.learner = learner

    def fit(self, chks, y=None):
        spectras = map(self.chk_fft, chks)
        fit_handling_iterables(self.learner, X=spectras, y=y)
        self.scalings_ = self.learner.scalings_
        return self


# TODO: Probably better if subclassing PCA
class TargettedVariancePCA(BaseEstimator, TransformerMixin):
    def __init__(self, target_variance=0.95, max_n_components=inf, min_n_components=1):
        self.target_variance = target_variance
        self.max_n_components = max_n_components
        self.min_n_components = min_n_components

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
        return matrix_mult(X, self.scalings_)


class SpectralProjectorUnsupervisedFitter(SpectralProjector, TargettedVariancePCA):
    def __init__(
        self,
        target_variance=0.95,
        max_n_components=inf,
        min_n_components=1,
        scalings_=None,
        chk_fft=DFLT_CHK_FFT,
    ):
        TargettedVariancePCA.__init__(
            self, target_variance, max_n_components, min_n_components
        )
        SpectralProjector.__init__(self, scalings_, chk_fft)

    def fit(self, chks, y=None):
        return super().fit(self.spectras(chks), y)


class MyLDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        lda.fit(X, ascertain_array(y))

        # Note: _lda not needed for the functioning of the class, but keeping around for debug reasons
        self._lda = lda

        self.scalings_ = lda.scalings_
        self.classes_ = lda.classes_
        return self

    def transform(self, X):
        return matrix_mult(X, self.scalings_)


class SpectralProjectorSupervisedFitter(SpectralProjector, MyLDA):
    def __init__(self, n_components=None, scalings_=None, chk_fft=DFLT_CHK_FFT):
        MyLDA.__init__(self, n_components)
        SpectralProjector.__init__(self, scalings_, chk_fft)

    def fit(self, chks, y):
        return super().fit(self.spectras(chks), y)


########################################################################################################################
# Utils to create and extend projections


import importlib
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

CLUSTERING_OPTIONS = (
    'KMeans',
    'SpectralClustering',
    'AffinityPropagation',
    'AgglomerativeClustering',
    'Birch',
    'MeanShift',
)


# Note: All these models (PCA, LDA, NCA, LinReg...) are learned with numpy.dot as matrix mult.
#   What happens if we learn with one definition of multiplication and run with another?

# TODO: Try changing all learn_spect_proj cases into subclasses of Projection and compare pros and cons.
# Note: Pro 1; Both cases and per-case arguments and defaults are visible, not hidden in code
# Note: Pro 2; Projection will wrap scalings_ and offer consistent methods for projection calculus
# Question: Do we want to keep scalings_ as the projecting matrix of Projection?
#   Both name and shape (transp) not consistent with pca, nca, linear regression, etc.

# TODO: Examples of objectivication below. Not verified or tested yet.
# Note: A choice to make here: Subclassing versus Delegation. I'll perform subclassing here.


class PcaProj(PCA, Projector):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.scalings_ = self.components_
        return self

    transform = Projector.transform


class IpcaProj(IncrementalPCA, Projector):
    def fit(self, X, y=None):
        super().fit(X, y)
        self.scalings_ = self.components_
        return self

    transform = Projector.transform


class LdaProj(LinearDiscriminantAnalysis, Projector):
    transform = Projector.transform


class LinregProj(LinearRegression, Projector):
    @wraps(LinearRegression.fit)
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.scalings_ = self.coef_.T
        return self


# Example of PCA+Projection, with delegation


class PcaProjWithDelegation(Projector):
    @wraps(PCA.__init__)
    def __init__(self, *args, **kwargs):
        self._source = PCA(*args, **kwargs)

    def fit(self, X, y=None):
        self._source.fit(X, y)
        self.scalings_ = self._source.components_
        return self

    transform = Projector.transform


def learn_spect_proj(
    X,
    y=None,
    spectral_proj_name='pca',
    clustering_meth='KMeans',
    clustering_options=CLUSTERING_OPTIONS,
    kwargs_feat=None,
    kwargs_clust=None,
):
    """
    Function to learn each of the important spectral projection

    :param X: the fvs, an array of size n*k
    :param y: the classes, an array of size n
    :param spectral_proj_name: a string of the name of the featurizer
    :param args: extra argument to be passed to the featurizer class
    :return: a matrix in the form of a numpy array
    """

    clustering_options = set(clustering_options)
    kwargs_feat = kwargs_feat or {'n_components': 10}
    kwargs_clust = kwargs_clust or {}

    assert (
        clustering_meth in clustering_options
    ), 'clustering options must one of {}'.format(
        ', '.join(map(str, clustering_options))
    )
    clusterer_m = getattr(importlib.import_module('sklearn.cluster'), clustering_meth)

    if spectral_proj_name == 'keep_features':
        indices = kwargs_feat['indices']
        proj_matrix = np.zeros((X.shape[1], len(indices)))
        for idx in range(len(indices)):
            proj_matrix[indices[idx], idx] = 1

    elif spectral_proj_name == 'pca':
        pca = PCA(**kwargs_feat)
        pca.fit(X)
        proj_matrix = pca.components_.T

    # elif spectral_proj_name == 'pseudo_pca':
    #     # make the pseudo pca proj matrix
    #     ppca = PseudoPca(**kwargs_feat)
    #     ppca.fit(X)
    #     proj_matrix = ppca.proj_mat.T

    elif spectral_proj_name == 'lda':
        lda = LDA(**kwargs_feat)
        lda.fit(X, y)
        n_components = kwargs_feat['n_components']
        proj_matrix = lda.scalings_[:, :n_components]

    elif spectral_proj_name == 'unsupervised_lda':
        n_components = kwargs_feat['n_components']
        if y is not None:
            print('y will be replaced by classes found by the chosen clusterer')
        if 'n_clusters' in clusterer_m.__init__.__code__.co_varnames:
            y = clusterer_m(n_clusters=n_components + 1, **kwargs_clust).fit_predict(X)
        else:
            y = clusterer_m(**kwargs_clust).fit_predict(X)
        lda = LDA(**kwargs_feat)
        lda.fit(X, y)
        proj_matrix = lda.scalings_[:, :n_components]

    elif spectral_proj_name == 'nca':
        nca = NCA(**kwargs_feat)
        nca.fit(X, y)
        proj_matrix = nca.components_.T

    elif spectral_proj_name == 'unsupervised_nca':
        n_components = kwargs_feat['n_components']
        if y is not None:
            print('y will be replaced by classes found by the chosen clusterer')
        if 'n_clusters' in clusterer_m.__init__.__code__.co_varnames:
            y = clusterer_m(n_clusters=n_components + 1, **kwargs_clust).fit_predict(X)
        else:
            y = clusterer_m(**kwargs_clust).fit_predict(X)
        nca = NCA(**kwargs_feat)
        nca.fit(X, y)
        proj_matrix = nca.components_.T

    elif spectral_proj_name == 'linear regression':
        lr = LinearRegression(**kwargs_feat)
        lr.fit(X, y)
        proj_matrix = lr.coef_.T

    else:
        all_spectral_proj = ', '.join(
            [
                'keep_features',
                'pca',
                'lda',
                'pseudo_pca',
                'unsupervised_lda',
                'unsupervised_nca',
                'nca',
                'linear regression',
            ]
        )
        raise ValueError(f'the spectral projector must be one of: {all_spectral_proj}')

    return proj_matrix


def residue(scalings, X):
    """
    find the residue of each of vectors after projection in basis

    residues will be vectors in the original space (same number of dimensions)

    :param scalings: an n-by-m array, spanning a vector space A
    :param X: an n-by-k array
    :return: an n-by-l array, the residues of vectors with the respect of the projection in basis

    >>> A = np.array([[1,0],[0,1]])
    >>> B = np.array([[2,3]])
    >>> print(residue(A, B))
    [[0 0]]
    >>> A = np.array([[1,0],[0,0]])
    >>> B = np.array([[2,3]])
    >>> print(residue(A, B))
    [[0 3]]
    """
    # assert X - projection(scalings, X) == X * (scalings - 1)
    return X - projection(scalings, X)


# TODO: Check. Not consistent with normal projection (scalings_ is not used as projection), though it is a linear trans.
# Note: Perhaps this is better seen as the compliment/dual of a projection?
class Residues(Projector):
    def transform(self, X):
        return X - projection(self.scalings_, X)


def keep_only_indices(indices, input_size=DFLT_INPUT_SIZE):
    """
    Makes a matrix which when a spectra is multiplied with it, only the entires in the list indices are kept
    :param indices: a list of indices to keep
    :param input_size: the total number of frequencies in the spectra
    :return: a matrix of size (input_size, len(indices))
    """

    proj_matrix = np.zeros((input_size, len(indices)))
    for idx in range(len(indices)):
        proj_matrix[indices[idx], idx] = 1
    return proj_matrix


def mk_pre_projection_from_indices(indices=None, input_size=DFLT_INPUT_SIZE):
    return keep_only_indices(indices, input_size=input_size)


def learn_chain_proj_matrix(
    X,
    y=None,
    chain=({'type': 'pca', 'args': {'n_components': 5}},),
    indices=None,
    input_size=1025,
):
    """
    A function successively learning a projections matrix on the residue of the previous one. The projections
    matrices are then concatenated and return as one single projection matrix. Note that the final projection
    matrix may not produce fvs of the size the sum of the components of each part, i.e., care must be taken
    to ensure each classes called must be able to produce the required number of components. For example,
    if the number of classes is 10, then lda can only produce 9 dimensions. To obtain say 12 dimension, the user
    will need to chain two lda's, for example with size 9 and 3 respectively.

    :param X: the fvs, an array of size n*k
    :param y: the classes, an array of size n
    :param chain: a tuple of dictionaries each containing the type of projection along with its parameters
    :param indices: the indices of the spectra to work with, anything else is discarded
    :param input_size: the total number of entries from the spectra.
        Only needed if n_freq is not None, in order to
        determine the size of the freq_selection_matrix
    :return: a single projection matrix
    """

    freq_selection_matrix = None
    if indices is not None:
        freq_selection_matrix = keep_only_indices(indices, input_size=input_size)
        X = np.dot(X, freq_selection_matrix)

    all_proj_matrices = []
    for mat_dict in chain:
        kwargs_feat = mat_dict['args']
        proj_matrix = learn_spect_proj(
            X, y, spectral_proj_name=mat_dict['type'], kwargs_feat=kwargs_feat
        )
        all_proj_matrices.append(proj_matrix)
        X = residue(proj_matrix, X)

    proj_matrix = np.hstack(tuple(all_proj_matrices))
    if indices is not None:
        proj_matrix = np.dot(freq_selection_matrix, proj_matrix)
    return np.array(proj_matrix)


def old_learn_chain_proj_matrix(
    X,
    y=None,
    chain=({'type': 'pca', 'kwargs': {'n_components': 5}},),
    indices=None,
    input_size=DFLT_INPUT_SIZE,
):
    freq_selection_matrix = None
    if indices is not None:
        freq_selection_matrix = keep_only_indices(indices, input_size=input_size)
        X = np.dot(X, freq_selection_matrix)

    all_proj_matrices = []
    for mat_dict in chain:
        kwargs_feat = mat_dict['kwargs']
        proj_matrix = learn_spect_proj(
            X, y, spectral_proj_name=mat_dict['type'], kwargs_feat=kwargs_feat
        )
        all_proj_matrices.append(proj_matrix)
        X = residue(proj_matrix, X)

    proj_matrix = np.hstack(tuple(all_proj_matrices))
    if indices is not None:
        proj_matrix = np.dot(freq_selection_matrix, proj_matrix)
    return np.array(proj_matrix)


class GeneralProjectionLearner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        chain=({'type': 'pca', 'args': {'n_components': 5}},),
        indices=None,
        n_freq=1025,
    ):
        self.chain = chain
        self.indices = indices
        self.n_freq = n_freq

    def fit(self, X, y=None):
        self.scalings_ = learn_chain_proj_matrix(
            X, y, self.chain, indices=self.indices, input_size=self.n_freq
        )
        self.projection = np.dot(self.scalings_, self.scalings_.T)
        return self

    def transform(self, X):
        """Projection within projected space (reduces dimensions)"""
        return np.dot(X, self.scalings_)

    def project(self, X):
        """Projection within original space (remains same dimensions, no reduction)"""
        return np.dot(X, self.projection)


def make_buckets(
    n_buckets=15,
    freqs_weighting=lambda x: x,
    freq_range=DFLT_FREQ_RANGE,
    non_empty_bucket=True,
    reverse=False,
):
    """
    Create greedily buckets starting by aggregating lower frequencies, when the sum of
    the frequencies values so far exceed the number of buckets created times the target
    average value for a single bucket, a new bucket is created with or without the last
    term according to which choice will be closest.

    :param n_buckets: final number of buckets
    :param freqs_weighting: any function assigning a non-negative value to each
        element in the freq_range.
        If given a list, the function will be assumed to be the one to one mapping the
        frequencies to the value in the list

    :param freq_range: the range of frequencies considered, inclusive on both ends
    :param non_empty_bucket: if set to true, all buckets will have at least one element
    :param reverse: if set to true, will start aggregating higher frequencies instead,
        which for an increasing weight
        function will aggregate the higher frequencies into larger bins as typical
    :return: a partition of freq_range

    >>> buckets = make_buckets(
    ... n_buckets=15,
    ... freqs_weighting=lambda x: np.log(x + 0.001),
    ... freq_range=(200, 1000))
    >>> len(buckets) == 15
    True
    >>> buckets[0][0] # the first bucket starts at the first term of freq_range
    200
    >>> buckets[-1][-1] # the last bucket ends at the last term of freq_range - 1
    999

    >>> make_buckets(3, freq_range=(0, 10))
    [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9]]
    >>> make_buckets(3, freqs_weighting=lambda x: x, freq_range=(0, 9))
    [[0, 1, 2, 3, 4, 5], [6, 7, 8]]
    >>> make_buckets(3, freqs_weighting=lambda x: 3, freq_range=(0, 9))
    [[0, 1, 2], [3, 4, 5, 6], [7, 8]]
    >>>
    >>> make_buckets(3, freqs_weighting=lambda x: 1 / (x + 1), freq_range=(0, 9))
    [[0], [1, 2], [3, 4, 5, 6, 7, 8]]
    >>> list(map(len, make_buckets(
    ... 8, freqs_weighting=lambda x: 1 / (x + 1), freq_range=(0, 1000)))
    ... )
    [1, 2, 5, 14, 34, 88, 225, 632]

    """

    # total number of frequencies
    low_freq = freq_range[0]
    high_freq = freq_range[1]
    n_freqs = high_freq - low_freq
    if n_freqs < n_buckets and non_empty_bucket:
        warnings.warn(
            'You asked for more buckets than the number of frequencies available, '
            'some will necessarily be empty'
        )
        non_empty_bucket = False

    # indices of the frequencies
    freq_range = list(range(low_freq, high_freq + 1))
    if reverse:
        freq_range.reverse()
    # get the value of each frequency
    if not hasattr(freqs_weighting, '__iter__'):
        freq_values = list(map(freqs_weighting, freq_range))
    else:
        freq_values = freqs_weighting
    # ideal average sum of bucket value
    target_bucket_sum = sum(freq_values) / n_buckets

    # position of the next term to consider
    position = 0
    # sum of all the existing buckets plus the current bucket under construction
    existing_bucket_sum = 0
    # list of indices for each bucket, empty at moment
    idx_bucket_list = []

    for idx in range(n_buckets):
        # the list of indices in the bucket under construction
        bucket_idx = []
        while existing_bucket_sum < target_bucket_sum * (idx + 1):
            bucket_idx.append(position)
            existing_bucket_sum += freq_values[position]
            # increment the position if we are not running out of terms
            if position < n_freqs - 1:
                position += 1
            # otherwise we stop and return all the buckets, including the one under construction
            # which is non empty by construction
            else:
                idx_bucket_list.append(bucket_idx)
                if low_freq > 0:
                    idx_bucket_list = [
                        [i + low_freq for i in l] for l in idx_bucket_list
                    ]
                return idx_bucket_list

        # if we skipped the loop above, our newly constructed bucket is empty
        # if we specified that empty buckets are not ok, we force one guy in it
        if non_empty_bucket and len(bucket_idx) == 0:
            bucket_idx.append(position)
            idx_bucket_list.append(bucket_idx)
            # increment the position if we are not running out of terms
            if position < n_freqs - 1:
                position += 1
            # otherwise we stop and return all the buckets, including the one under construction
            # which is non empty by construction
            else:
                return idx_bucket_list

        else:
            # we now have a non empty bucket, so we check what is best, including the last added term or no
            total_dif_small = abs(
                existing_bucket_sum
                - freq_values[position]
                - target_bucket_sum * (idx + 1)
            )
            total_diff_large = abs(existing_bucket_sum - target_bucket_sum * (idx + 1))

            # we don't remove the last if it is better not too or if we want non empty buckets
            # and remove it would violate that rule
            if total_diff_large < total_dif_small or (
                len(bucket_idx) < 2 and non_empty_bucket
            ):
                idx_bucket_list.append(bucket_idx)
            # otherwise remove last term
            else:
                idx_bucket_list.append(bucket_idx[:-1])
                # below is to ensure that if the last bucket is empty, we increment the position
                # effectively skipping that position. Otherwise we will end up with a list of empty lists
                if len(bucket_idx[:-1]) > 0:
                    existing_bucket_sum -= freq_values[position]
                    position -= 1
                else:
                    pass

    idx_bucket_list[-1] = freq_range[idx_bucket_list[-1][0] :]
    if low_freq > 0:
        idx_bucket_list = [[i + low_freq for i in l] for l in idx_bucket_list]

    return idx_bucket_list


def _plus_one_inverse(x):
    return 1 / (x + 1)


def frequency_weight_based_bands_matrix(
    n_buckets=15, freqs_weighting=_plus_one_inverse, n_freq=1025, reverse=False
):
    """
    Get a projection matrix of disjoint bands based on a frequency weighting function.

    >>> assert frequency_weight_based_bands_matrix(3, n_freq=9).tolist() == [
    ... [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ... [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ... [0.0, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    ... ]

    """
    buckets = make_buckets(
        n_buckets, freqs_weighting, freq_range=(0, n_freq + 1), reverse=reverse
    )
    return make_band_matrix(buckets, n_freq=n_freq)


# class GeneralProjectionLearner(BaseEstimator, TransformerMixin):
#     def __init__(self, chain=({'type': 'pca', 'kwargs': {'n_components': 5}},), indices=None,
#                  input_size=DFLT_INPUT_SIZE):
#         self.chain = chain
#         self.indices = indices
#         self.input_size = input_size
#
#     def fit(self, X, y=None):
#         pre_projection = mk_pre_projection_from_indices(self.indices, self.input_size)
#         self.scalings_ = learn_chain_proj_matrix(X, y, self.chain, pre_projection)
#         return self
#
#     def fit(self, X, y=None):
#         self.scalings_ = learn_chain_proj_matrix(X, y, self.chain, indices=self.indices, n_freq=self.n_freq)
#         self.projection = np.dot(self.scalings_, self.scalings_.T)
#         return self
#
#     def transform(self, X):
#         """Projection within projected space (reduces dimensions)"""
#         return np.dot(X, self.scalings_)
#
#     def project(self, X):
#         """Projection within original space (remains same dimensions, no reduction)"""
#         return np.dot(X, self.projection)
#
