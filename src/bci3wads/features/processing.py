import pickle
import pathlib
import numpy as np

from bci3wads.utils import constants


class Epoch:
    def __init__(self, signals, flashing, stimulus_codes, stimulus_types,
                 target_char):
        self.n_channels = signals.shape[1]
        self.signals = signals
        self.flashing = flashing
        self.stimulus_codes = stimulus_codes
        self.stimulus_types = stimulus_types
        self.target_char = target_char

    def flash_start_indices(self):
        indices = [
            i for i in range(len(self.flashing))
            if (i == 0)  # Each epoch begins with the first flash
            or (self.flashing[i] == 1 and self.flashing[i - 1] == 0)
        ]

        return indices

    def sample_channel(self, indices, channel_id=constants.CHANNEL_ID,
                       window_size=constants.WINDOW_SIZE):
        channel_signals = self.signals[:, channel_id]

        samples = np.array([
            channel_signals[i:i+window_size]
            for i in indices
        ])

        return samples

    def samples_codes(self, indices):
        return self.stimulus_codes[indices]

    def process_channel(self, channel_id=constants.CHANNEL_ID,
                        window_size=constants.WINDOW_SIZE):
        indices = self.flash_start_indices()
        samples = self.sample_channel(indices, window_size=window_size,
                                      channel_id=channel_id)
        codes = self.samples_codes(indices)

        n_codes = len(np.unique(codes))  # Should be 12

        positions = np.array([
            np.nonzero(codes == i)[0]
            for i in range(n_codes)
        ])

        processed = np.array([samples[position] for position in positions])

        return processed

    def process_channels(self, channel_ids=constants.CHANNEL_IDS,
                         window_size=constants.WINDOW_SIZE):
        processed_channels = np.concatenate([
            self.process_channel(channel_id, window_size)
            for channel_id in channel_ids
        ], axis=-1)

        return processed_channels


class Subject:
    def __init__(self, filename):
        with open(constants.INTER_DATA_PATH.joinpath(filename), 'rb') as f:
            data = pickle.load(f)

        self.name = pathlib.Path(filename).stem
        self.signals = data['signals']
        self.target_chars = data['target_chars']
        self.flashings = data['flashings']
        self.stimulus_codes = data['stimulus_codes']
        self.stimulus_types = data['stimulus_types']
        self.epochs = [
            Epoch(signal, flashing, codes, types, target_char)
            for signal, flashing, codes, types, target_char in zip(
                self.signals, self.flashings, self.stimulus_codes,
                self.stimulus_types, self.target_chars
            )
        ]

    def process_epoch(self, epoch_id, channel_ids=constants.CHANNEL_IDS,
                      window_size=constants.WINDOW_SIZE):
        processed_channels = self.epochs[epoch_id].process_channels(
            channel_ids, window_size)

        return processed_channels
