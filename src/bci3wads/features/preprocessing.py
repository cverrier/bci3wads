import pickle
import pathlib
import scipy.io

from bci3wads.utils import constants


class Subject:
    def __init__(self, filename, is_train=True):
        data = scipy.io.loadmat(constants.RAW_DATA_PATH.joinpath(filename))

        self.is_train = is_train
        self.name = pathlib.Path(filename).stem
        self.signals = data['Signal']
        self.target_chars = data['TargetChar'] if is_train else None
        self.flashings = data['Flashing']
        self.stimulus_codes = data['StimulusCode']
        self.stimulus_types = data['StimulusType'] if is_train else None

    def clean_target_chars(self):
        target_chars = [
            char for char in self.target_chars[0]
        ]
        return target_chars

    def clean_flashings(self):
        return self.flashings.astype(int)

    def clean_stimulus_codes(self):
        return self.stimulus_codes.astype(int) - 1

    def clean_stimulus_types(self):
        return self.stimulus_types.astype(bool)

    def clean_data(self):
        cleaned = {}

        cleaned['signals'] = self.signals

        if self.is_train:
            cleaned['target_chars'] = self.clean_target_chars()
            cleaned['stimulus_types'] = self.clean_stimulus_types()

        cleaned['flashings'] = self.clean_flashings()
        cleaned['stimulus_codes'] = self.clean_stimulus_codes()

        return cleaned

    def save(self, data):
        filename = self.name + '.pickle'
        data_path = constants.INTER_DATA_PATH.joinpath(filename)

        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
