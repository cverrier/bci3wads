import numpy as np
import matplotlib.pyplot as plt

from bci3wads.utils import data
from bci3wads.utils import constants
from bci3wads.features.epoch import Epoch

subject_match_signals = []
subject_mismatch_signals = []

for epoch_path in constants.REF_SUBJECT_PATH.glob('epoch_*.pickle'):
    epoch = Epoch(data.load_pickle(epoch_path))

    match_signals = epoch.get_match_signals()
    mismatch_signals = epoch.get_mismatch_signals()

    subject_match_signals.append(match_signals)
    subject_mismatch_signals.append(mismatch_signals)

subject_match_signals = np.concatenate(subject_match_signals)
subject_mismatch_signals = np.concatenate(subject_mismatch_signals)

match_average = np.mean(subject_match_signals, axis=(0, 1))
mismatch_average = np.mean(subject_mismatch_signals, axis=(0, 1))

plt.plot(match_average, label='match')
plt.plot(mismatch_average, label='mismatch')
plt.legend()
plt.show()
