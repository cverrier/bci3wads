import pickle
import pathlib
import numpy as np
import bci3wads.models as models

from bci3wads.utils import data
from bci3wads.utils import constants
from bci3wads.features.epoch import Epoch
from bci3wads.features.estimator import Estimator
from bci3wads.features.formatter import Formatter


subject_name = 'Subject_A_Train'
channels_tag = 'channels_11'
model_name = 'improved_detector'

subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
    subject_name / channels_tag

results_path = constants.PREDICTIONS_PATH / \
    model_name / subject_name / channels_tag
results_path.mkdir(parents=True, exist_ok=True)

if model_name == 'basic_detector':
    target_signal = np.zeros(constants.WINDOW_SIZE)
    target_signal[(300 * 240) // 1000:(500 * 240) // 1000] = 1.0

    noise_cov = np.eye(len(target_signal))
elif model_name == 'improved_detector':
    estimator = Estimator(subject_path)
    target_signal = estimator.estimate_target_signal()
    noise_cov = estimator.estimate_noise_cov()
else:
    raise NotImplementedError()

for n_trials in range(1, constants.N_TRIALS + 1):
    predicted_chars = []

    for epoch_path in sorted(list(subject_path.glob('epoch_*.pickle')),
                             key=lambda p: int(p.stem.split('_')[-1])):
        epoch = Epoch(data.load_pickle(epoch_path))

        trials_inds = np.arange(n_trials)
        signals = epoch.average_signals(trials_inds=trials_inds)

        detector = models.detector.Detector()
        detector.fit(target_signal, noise_cov=noise_cov)

        scores = detector.predict(signals)

        formatter = Formatter(scores)
        predicted_char = formatter.get_char()

        predicted_chars.append(predicted_char)

    filename = f'predictions_{n_trials}_trials.npy'
    np.save(results_path.joinpath(filename), predicted_chars)
