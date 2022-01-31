import numpy as np


class Epoch:
    def __init__(self, data):
        self.target_char = data.get('target_char')
        self.target_char_codes = data.get('target_char_codes')
        self.target_char_coords = data.get('target_char_coords')
        self.epoch_id = data['epoch_id']
        self.channel_ids = data['channel_ids']
        self.signals = data['processed_channels']

    def get_match_signals(self):
        return self.signals[self.target_char_codes]

    def get_mismatch_signals(self):
        mismatch_codes = [
            code for code in range(len(self.signals))
            if code not in self.target_char_codes
        ]

        return self.signals[mismatch_codes]

    def average_signals(self, cls=None, trials_inds=None):
        if trials_inds is None:
            n_trials = self.signals.shape[1]
            trials_inds = np.arange(n_trials)

        if cls == 'match':
            signals = self.get_match_signals()[:, trials_inds, :]
            average = np.mean(signals, axis=(0, 1))
        elif cls == 'mismatch':
            signals = self.get_mismatch_signals()[:, trials_inds, :]
            average = np.mean(signals, axis=(0, 1))
        else:
            signals = self.signals[:, trials_inds, :]
            average = np.mean(signals, axis=1)

        return average

    pass
