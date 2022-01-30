import numpy as np
import bci3wads.models as models

from bci3wads.utils import constants

seed = 42  # For reproducible results
n_obs = 100
n_match_obs = 5
snr = 3.0

rng = np.random.default_rng(seed=seed)

match_signals_inds = rng.choice(n_obs, size=n_match_obs, replace=False)

target_signal = np.zeros(constants.WINDOW_SIZE)
window = np.arange(len(target_signal) // 3, len(target_signal) // 2)
target_signal[window] = 1.0

amplitude = (
    snr * np.sqrt(constants.WINDOW_SIZE) / np.linalg.norm(target_signal)
)

noise_mean = np.zeros(len(target_signal))
noise_cov = np.eye(len(target_signal))

signals = rng.multivariate_normal(mean=noise_mean, cov=noise_cov, size=n_obs)
signals[match_signals_inds] += amplitude * target_signal

detector = models.detector.Detector()
detector.fit(target_signal, noise_cov=noise_cov)

scores = detector.predict_scores(signals)
predictions = detector.predict(signals, n_match=n_match_obs)

print(f'Match indices: {np.sort(match_signals_inds)}')
print(f'Scores: {scores}')
print(f'Predictions: {predictions}')
print('Predicted match indices:',
      np.sort(np.argsort(predictions)[-n_match_obs:]))
