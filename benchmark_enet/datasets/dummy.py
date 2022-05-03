from benchopt import BaseDataset
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = 'Dummy dataset'
    parameters = {
        'n_samples': [30],
        'n_features': [50],
    }

    def __init__(self, n_samples, n_features, random_state=0):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        n_samples, n_features = self.n_samples, self.n_features
        np.random.seed(self.random_state)

        X = np.random.randn(n_samples, n_features)
        Y = np.random.randn(n_samples)
        return n_samples, {'X': X, 'y': Y}
