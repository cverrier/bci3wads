import numpy as np
import sklearn.base


class CovEstimator(sklearn.base.BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.covariance_ = np.cov(X, rowvar=False)

        return self
