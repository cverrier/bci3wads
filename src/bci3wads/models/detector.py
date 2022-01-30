import numpy as np
import scipy.linalg


class Detector:
    def __init__(self):
        pass

    def fit(self, target_signal, noise_cov=None, inv_noise_cov=None):
        if inv_noise_cov is None:
            v = scipy.linalg.solve(noise_cov, target_signal, assume_a='pos')
        else:
            v = inv_noise_cov @ target_signal

        norm = np.linalg.norm(v)

        self.coef_ = v / norm

    def predict_scores(self, signals):
        return signals @ self.coef_

    def predict(self, signals, n_match=2):
        scores = self.predict_scores(signals)
        best_scores_inds = np.argsort(scores)[-n_match:]
        predictions = np.zeros(len(signals), dtype=bool)
        predictions[best_scores_inds] = True

        return predictions

    pass
