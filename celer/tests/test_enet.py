import pytest
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_array_less
from sklearn.linear_model import (enet_path, ElasticNet as sk_ElasticNet, lasso_path)

from celer import Lasso, ElasticNet, celer_path
from celer.utils.testing import build_dataset


def test_raise_errors_l1_ratio():
    with np.testing.assert_raises(ValueError):
        ElasticNet(l1_ratio=5.)

    with np.testing.assert_raises(NotImplementedError):
        X, y = build_dataset(n_samples=30, n_features=50)
        y = np.sign(y)
        celer_path(X, y, 'logreg', l1_ratio=0.5)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_enet_lasso_equivalence(sparse_X):
    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    coef_lasso = Lasso().fit(X, y).coef_
    coef_enet = ElasticNet(l1_ratio=1.0).fit(X, y).coef_

    assert_allclose(coef_lasso, coef_enet)


@pytest.mark.parametrize("sparse_X, prune", [(False, 0)])
def test_celer_enet_sk_enet_equivalence(sparse_X, prune):
    """Test that celer_path matches sklearn lasso_path."""

    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    tol = 1e-14
    l1_ratio = 0.7
    n_alphas = 10
    alpha_max = norm(X.T @ y, ord=np.inf) / (n_samples * l1_ratio)
    params = dict(eps=1e-3, n_alphas=n_alphas, tol=tol, l1_ratio=l1_ratio)

    alphas1, coefs1, gaps1 = celer_path(
        X, y, "lasso", return_thetas=False, verbose=0, prune=prune,
        max_iter=30, **params)

    alphas2, coefs2, _ = enet_path(X, y, verbose=0, **params,
                                   max_iter=10000)

    assert_allclose(alphas1, alphas2)
    assert_array_less(gaps1, tol * norm(y) ** 2 / n_samples)
    assert_allclose(coefs1, coefs2, rtol=1e-03, atol=1e-4)


if __name__ == '__main__':
    test_celer_enet_sk_enet_equivalence(False, 0)