# Authors:
# Mathurin Massias
# Thomas Moreau

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state
from scipy.linalg import block_diag


def make_correlated_data(n_samples=100, n_features=50, corr=0.6, snr=3,
                         density=0.2, w_true=None, random_state=None):
    r"""Generate correlated design matrix with decaying correlation rho**|i-j|.
    according to

    .. math::

        y = X w^* + \epsilon

    such that :math:`||X w^*|| / ||\epsilon|| = snr`.

    The generated features have mean 0, variance 1 and the expected correlation
    structure:

    .. math::

        \mathbb E[x_i] = 0~, \quad \mathbb E[x_i^2] = 1  \quad
        \text{and} \quad \mathbb E[x_ix_j] = \rho^{|i-j|}


    Parameters
    ----------
    n_samples: int
        Number of samples in the design matrix.
    n_features: int
        Number of features in the design matrix.
    corr: float
        Correlation :math:`\rho` between successive features. The element
        :math:`C_{i, j}` in the correlation matrix will be
        :math:`\rho^{|i-j|}`. This parameter should be selected in
        :math:`[0, 1[`.
    snr: float or np.inf
        Signal-to-noise ratio. In np.inf, no noise is added.
    density: float
        Proportion of non zero elements in w_true if it must be simulated.
    w_true: np.array, shape (n_features,) | None
        True regression coefficients. If None, an array with `nnz` non zero
        standard Gaussian entries is simulated.
    random_state: int | RandomState instance | None (default)
        Determines random number generation for data generation. Use an int to
        make the randomness deterministic.

    Returns
    -------
    X: ndarray, shape (n_samples, n_features)
        A design matrix with Toeplitz covariance.
    y: ndarray, shape (n_samples,)
        Observation vector.
    w_true: ndarray, shape (n_features,)
        True regression vector of the model.
    """
    if not 0 <= corr < 1:
        raise ValueError("The correlation `corr` should be chosen in [0, 1[.")
    if not 0 < density <= 1:
        raise ValueError("The density should be chosen in ]0, 1].")
    rng = check_random_state(random_state)
    nnz = int(density * n_features)

    if corr == 0:
        X = np.asfortranarray(rng.randn(n_samples, n_features))
    else:
        # X is generated cleverly using an AR model with reason corr and
        # innovation sigma^2 = 1 - corr ** 2: X[:, j+1] = corr X[:, j] + eps_j
        # where eps_j = sigma * rng.randn(n_samples)
        sigma = np.sqrt(1 - corr ** 2)
        U = rng.randn(n_samples)

        X = np.empty([n_samples, n_features], order='F')
        X[:, 0] = U
        for j in range(1, n_features):
            U *= corr
            U += sigma * rng.randn(n_samples)
            X[:, j] = U

    if w_true is None:
        w_true = np.zeros(n_features)
        support = rng.choice(n_features, nnz, replace=False)
        w_true[support] = rng.randn(nnz)

    y = X @ w_true
    if snr != np.inf:
        noise = rng.randn(n_samples)
        y += noise / norm(noise) * norm(y) / snr
    return X, y, w_true


def make_group_correlated_data(n_samples: int, n_features: int, n_groups: int,
                               corr: float = 0.8, snr: float = 3,
                               return_w_group: bool = False, random_state: int = None):
    r"""Generate design matrix with group correlated features.

    .. math::

        y = X w^* + \epsilon

    such that :math:`||X w^*|| / ||\epsilon|| = snr`.

    The row of X are drawn from a multi-variate normal distribution with mean 0
    and n_group-bloc diagonal covariance matrix. 
    Every bloc B is a n_features // n_groups square matrix where

    .. math::

        B[i, j] = 
        \begin{cases}
            corr & \text{if} i \neq j \\
            0 & \text{otherwise} \\
        \end{cases}

    Parameters
    ----------
    n_samples : int
        Number of samples in the design matrix.
    n_features : int
        Number of features in the design matrix.
    n_groups : int
        Number of groups of correlated features. Also the number
        of blocs in correlation matrix.
    corr : float, optional
        Correlation whithin a group of features, by default 0.8.
        Must be chosen in [0, 1).
    snr : float, or np.inf optional
        Signal to noise ratio, by default 3. When ``np.inf`` is provided,
        no noise is added.
    random_state : int, optional
        Number to control the randomness of data generation, by default None.
    """
    if not 0 <= corr < 1:
        raise ValueError(
            "The correlation `corr` must be chosen in [0, 1). "
            "got: %r" % corr)

    if random_state is not None:
        np.random.seed(random_state)

    n_features_group = n_features // n_groups

    # build corr matrix
    blocs_corr_matrix = []
    for g in range(n_groups):
        bloc_matrix = np.array([[corr]*i + [1] + [corr]*(n_features_group-i-1)
                                for i in range(n_features_group)], dtype=float)
        blocs_corr_matrix.append(bloc_matrix)

    corr_matrix = block_diag(*blocs_corr_matrix)

    # weight vector
    w_group = np.random.choice(2, n_groups)
    w_true = np.repeat(w_group, n_features_group)

    # build X, y
    mean_vec = np.zeros(n_features)
    X = np.random.multivariate_normal(mean_vec, corr_matrix, size=n_samples)
    y = X @ w_true

    if snr != np.inf:
        noise = np.random.randn(n_samples)
        y += noise / norm(noise) * norm(y) / snr

    return X, y, w_true
