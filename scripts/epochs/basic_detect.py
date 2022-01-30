import pickle
import numpy as np
import bci3wads.models as models

from bci3wads.features.epoch import Epoch
from bci3wads.utils import constants

with open(constants.REF_EPOCH_PATH, 'rb') as f:
    data = pickle.load(f)

epoch = Epoch(data)

target_signal = np.zeros(constants.WINDOW_SIZE)
target_signal[(300 * 240) // 1000:(500 * 240) // 1000] = 1.0

noise_cov = np.eye(len(target_signal))

signals = epoch.average_signals()

detector = models.detector.Detector()

detector.fit(target_signal, noise_cov=noise_cov)

predictions = np.argsort(detector.predict(signals))[-2:]
