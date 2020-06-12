from functools import wraps
import inspect

from numpy import cumsum, min, inf, ndarray, floor
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
from numpy import array, hanning, fft  # TODO: Get rid of this once we use C-based spectr

DFLT_WIN_FUNC = hanning
DFLT_CHK_SIZE = 2048
DFLT_INPUT_SIZE = 1 + int(floor(DFLT_CHK_SIZE / 2))
DFLT_AMPLITUDE_FUNC = np.abs


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

matrix_mult = np.dot
DFLT_MATRIX_MULTI = matrix_mult


# TODO: Does the doc match what the function does?
# TODO: Better name for this, to distinguish between expressing projection in projected sub-space or original space
#   The following expresses it in the original space
def projection(basis, vectors):
    """
    The vectors live in a k dimensional space S and the columns of the basis are vectors of the same
    space spanning a subspace of S. Gives a representation of the projection of vector into the space
    spanned by basis in term of the basis.

    :param basis: an n-by-k array, a matrix whose vertical columns are the vectors of the basis
    :param vectors: an m-by-k array, a vector to be represented in the basis
    :return: an m-by-k array
    """
    return matrix_mult(matrix_mult(vectors, basis), basis.T)


def reducing_proj(basis, vectors):
    """What we actually use to get fvs from spectras"""
    return matrix_mult(vectors, basis.T)


class Projector:
    def __init__(self, scalings_=None, mat_mult=DFLT_MATRIX_MULTI):
        self.scalings_ = scalings_
        self.mat_mult = mat_mult

    def transform(self, X):
        # return projection(self.scalings_, ascertain_array(X))
        return reducing_proj(self.scalings_, ascertain_array(X))

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
            raise NotFittedError("{} was not fitted yet.".format(self.__class__.__name__))


class SpectralProjector(Projector):
    def __init__(self,
                 scalings_=None,
                 chk_fft=DFLT_CHK_FFT,
                 mat_mult=DFLT_MATRIX_MULTI):
        super().__init__(scalings_, mat_mult)
        self.chk_fft = chk_fft
        self.chk_size = getattr(chk_fft, 'chk_size', None)  # get chk_size from chk_fft if it has it

    def spectras(self, chks):
        return array([self.chk_fft(chk) for chk in chks])

    def transform(self, chks):
        return super().transform(self.spectras(chks))

    def __call__(self, chk):
        return self.transform([chk])[0]

    mk_chk_fft = staticmethod(mk_chk_fft)  # to have it available to make a chk_fft in __init__

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
    def __init__(self,
                 learner,
                 chk_fft=DFLT_CHK_FFT,
                 mat_mult=DFLT_MATRIX_MULTI):
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
    def __init__(self, target_variance=0.95, max_n_components=inf, min_n_components=1,
                 scalings_=None, chk_fft=DFLT_CHK_FFT):
        TargettedVariancePCA.__init__(self, target_variance, max_n_components, min_n_components)
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
    def __init__(self, n_components=None,
                 scalings_=None, chk_fft=DFLT_CHK_FFT):
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

CLUSTERING_OPTIONS = ('KMeans', 'SpectralClustering', 'AffinityPropagation',
                      'AgglomerativeClustering', 'Birch', 'MeanShift')


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


def learn_spect_proj(X, y=None, spectral_proj_name='pca',
                     clustering_meth='KMeans',
                     clustering_options=CLUSTERING_OPTIONS,
                     kwargs_feat=None,
                     kwargs_clust=None):
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

    assert clustering_meth in clustering_options, 'clustering options must one of {}'.format(
        ', '.join(map(str, clustering_options)))
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
        all_spectral_proj = ', '.join(['keep_features', 'pca',
                                       'lda', 'pseudo_pca',
                                       'unsupervised_lda',
                                       'unsupervised_nca',
                                       'nca',
                                       'linear regression'])
        raise ValueError(f'the spectral projector must be one of: {all_spectral_proj}')

    return proj_matrix


def residue(scalings, X):
    """
    find the residue of each of vectors after projection in basis
    :param scalings: an n-by-k array, spanning a vector space A
    :param X: an n-by-l array
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


def learn_chain_proj_matrix(X, y=None, chain=({'type': 'pca', 'kwargs': {'n_components': 5}},),
                            pre_projection=None):
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
    :param input_size: the total number of entries from the spectra. Only needed if input_size is not None, in order to
                   determine the size of the freq_selection_matrix
    :return: a single projection matrix
    """

    if pre_projection is not None:
        X = np.dot(X, pre_projection)

    all_proj_matrices = []
    for mat_dict in chain:
        kwargs_feat = mat_dict['kwargs']
        proj_matrix = learn_spect_proj(X, y,
                                       spectral_proj_name=mat_dict['type'],
                                       kwargs_feat=kwargs_feat)
        all_proj_matrices.append(proj_matrix)
        X = residue(proj_matrix, X)

    proj_matrix = np.hstack(tuple(all_proj_matrices))
    if pre_projection is not None:  # if you had a prior matrix, need to combine with the proj_matrix
        proj_matrix = np.dot(pre_projection, proj_matrix)
    return np.array(proj_matrix)


def old_learn_chain_proj_matrix(X, y=None, chain=({'type': 'pca', 'kwargs': {'n_components': 5}},),
                                indices=None, input_size=DFLT_INPUT_SIZE):
    freq_selection_matrix = None
    if indices is not None:
        freq_selection_matrix = keep_only_indices(indices, input_size=input_size)
        X = np.dot(X, freq_selection_matrix)

    all_proj_matrices = []
    for mat_dict in chain:
        kwargs_feat = mat_dict['kwargs']
        proj_matrix = learn_spect_proj(X, y,
                                       spectral_proj_name=mat_dict['type'],
                                       kwargs_feat=kwargs_feat)
        all_proj_matrices.append(proj_matrix)
        X = residue(proj_matrix, X)

    proj_matrix = np.hstack(tuple(all_proj_matrices))
    if indices is not None:
        proj_matrix = np.dot(freq_selection_matrix, proj_matrix)
    return np.array(proj_matrix)


class GeneralProjectionLearner(BaseEstimator, TransformerMixin):
    def __init__(self, chain=({'type': 'pca', 'kwargs': {'n_components': 5}},), indices=None,
                 input_size=DFLT_INPUT_SIZE):
        self.chain = chain
        self.indices = indices
        self.input_size = input_size

    def fit(self, X, y=None):
        pre_projection = mk_pre_projection_from_indices(self.indices, self.input_size)
        self.scalings_ = learn_chain_proj_matrix(X, y, self.chain, pre_projection)
        return self

    def transform(self, X):
        return np.dot(X, self.scalings_)
