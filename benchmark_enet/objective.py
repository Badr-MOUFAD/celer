from benchopt.base import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Objective(BaseObjective):
    """
    ElasticNet objective function

        (1 / (2 * n_samples)) * ||y - X w||^2_2
        + alpha * l1_ratio *  ||w_j||_1
        + 0.5 * alpha * (1 - l1_ratio) *  ||w_j||^2_2)
    """
    name = "ElasticNet regression"
    parameters = {
        'frac_alpha_max': [1e-2],
        'l1_ratio': [.5],
    }

    def __init__(self, frac_alpha_max, l1_ratio):
        self.frac_alpha_max = frac_alpha_max
        self.l1_ratio = l1_ratio

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples = X.shape[0]
        self.alpha = self.frac_alpha_max * self._get_alpha_max()

    def compute(self, beta):
        X, y = self.X, self.y
        n_samples = self.n_samples
        alpha, l1_ratio = self.alpha, self.l1_ratio

        quad_term = 0.5 / n_samples * norm(y - X @ beta, ord=2)**2
        penalty_term = alpha * l1_ratio * norm(beta, ord=1) + \
            0.5 * alpha * (1 - l1_ratio) * norm(beta, ord=2)**2

        return quad_term + penalty_term

    def to_dict(self):
        X, y = self.X, self.y
        alpha, l1_ratio = self.alpha, self.l1_ratio

        return {'X': X, 'y': y, 'alpha': alpha, 'l1_ratio': l1_ratio}

    def _get_alpha_max(self,):
        X, y = self.X, self.y
        return norm(X.T @ y, ord=np.inf)
