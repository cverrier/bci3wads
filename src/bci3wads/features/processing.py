from importlib.resources import path
import pickle
import pathlib

from bci3wads.utils import constants


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

    def flashing_start_indices(self, epoch_id):
        flashing = self.flashings[epoch_id]
        indices = [
            i for i in range(len(flashing))
            if (i == 0)  # Each epoch begins with the first flash
            or (flashing[i] == 1 and flashing[i - 1] == 0)
        ]

        return indices

    pass
