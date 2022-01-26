from distutils.command.clean import clean
import pathlib
import scipy.io
import numpy as np


class Subject:
    def __init__(self, data_path: pathlib.Path):
        data = scipy.io.loadmat(data_path)

        self.signals = data['Signal']
        self.target_chars = data['TargetChar']
        self.flashings = data['Flashing']
        self.stimulus_codes = data['StimulusCode']
        self.stimulus_types = data['StimulusType']

    def clean_target_chars(self):
        target_chars = self.target_chars = [
            char for char in self.target_chars[0]
        ]
        return target_chars

    def clean_flashings(self):
        return self.flashings.astype(int)

    def clean_stimulus_codes(self):
        return self.stimulus_types.astype(int) - 1

    def clean_stimulus_types(self):
        return self.stimulus_types.astype(bool)

    def clean_data(self):
        cleaned = {}

        cleaned['signals'] = self.signals
        cleaned['target_chars'] = self.clean_target_chars()
        cleaned['flashings'] = self.clean_flashings()
        cleaned['stimulus_codes'] = self.clean_stimulus_codes()
        cleaned['stimulus_types'] = self.clean_stimulus_types()

        return cleaned
