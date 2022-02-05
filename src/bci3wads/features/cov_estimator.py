import numpy as np
import sklearn.base
import pathlib

from bci3wads.features.estimator import Estimator


class CovEstimator(sklearn.base.BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.covariance_ = np.cov(X, rowvar=False)
        # subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
        #     'Subject_A_Train' / 'channels_11'
        # estimator = Estimator(subject_path)

        # self.covariance_ = estimator.estimate_noise_cov(method='mismatch')

        return self
