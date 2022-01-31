import pickle
import pathlib
import numpy as np
import bci3wads.models as models

from bci3wads.features.epoch import Epoch
from bci3wads.features.estimator import Estimator
from bci3wads.utils import constants
from bci3wads.features.formatter import Formatter


subject_name = 'Subject_B_Train'
channels_tag = 'channels_11'

subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
    subject_name / channels_tag

results_path = constants.PREDICTIONS_PATH / \
    'improved_detector' / subject_name / channels_tag
results_path.mkdir(parents=True, exist_ok=True)

# target_signal = np.zeros(constants.WINDOW_SIZE)
# target_signal[(300 * 240) // 1000:(500 * 240) // 1000] = 1.0

# noise_cov = np.eye(len(target_signal))

estimator = Estimator(subject_path)
target_signal = estimator.estimate_target_signal()
noise_cov = estimator.estimate_noise_cov()

predicted_chars = []

for epoch_path in sorted(list(subject_path.glob('epoch_*.pickle')),
                         key=lambda p: int(p.stem.split('_')[-1])):
    with open(epoch_path, 'rb') as f:
        data = pickle.load(f)

    epoch = Epoch(data)

    signals = epoch.average_signals()

    detector = models.detector.Detector()
    detector.fit(target_signal, noise_cov=noise_cov)

    scores = detector.predict(signals)

    formatter = Formatter(scores)
    predicted_char = formatter.get_char()

    predicted_chars.append(predicted_char)

np.save(results_path.joinpath('predictions.npy'), predicted_chars)
