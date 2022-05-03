import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from sklearn.linear_model import ElasticNet
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'sklearn ElasticNet'
    parameters = {
        'fit_intercept': [False],
        'tol': [1e-14, 1e-13, 1e-12],
    }

    def __init__(self, fit_intercept, tol):
        self.fit_intercept = fit_intercept
        self.tol = tol

    def set_objective(self, X, y, alpha, l1_ratio):
        self.X, self.y = X, y

        self.reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                              tol=self.tol, fit_intercept=self.fit_intercept)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    def run(self, n_iter):
        X, y = self.X, self.y
        self.reg.max_iter_ = n_iter
        self.reg.fit(X, y)

    def get_result(self):
        return self.reg.coef_
