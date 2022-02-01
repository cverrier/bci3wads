import numpy as np

from bci3wads.utils import data
from bci3wads.features.epoch import Epoch


class Estimator:
    def __init__(self, epochs_path):
        epochs = []
        for epoch_path in epochs_path.glob('epoch_*.pickle'):
            epoch = Epoch(data.load_pickle(epoch_path))
            epochs.append(epoch)

        self.epochs = epochs

    def get_match_proba(self):
        epoch = self.epochs[0]

        return len(epoch.target_char_codes) / len(epoch.signals)

    def get_mismatch_proba(self):
        return 1 - self.get_match_proba()

    def estimate_target_signal(self, method='match_avg'):
        match_signals = np.concatenate(
            [epoch.get_match_signals() for epoch in self.epochs]
        )
        match_avg = np.mean(match_signals, axis=(0, 1))

        if method == 'match_avg':
            return match_avg

        elif method in ['match_mismatch_avg_diff', 'mismatch_avg']:
            mismatch_signals = np.concatenate(
                [epoch.get_mismatch_signals() for epoch in self.epochs]
            )
            mismatch_avg = np.mean(mismatch_signals, axis=(0, 1))

            if method == 'match_mismatch_avg_diff':
                return match_avg - mismatch_avg
            else:
                return mismatch_avg
        else:
            raise NotImplementedError()

    def estimate_noise_cov(self):
        mismatch_signals = np.concatenate(
            [epoch.get_mismatch_signals() for epoch in self.epochs]
        )
        n_vars = mismatch_signals.shape[-1]

        return np.cov(mismatch_signals.reshape(-1, n_vars), rowvar=False)
