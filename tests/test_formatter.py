import pytest
import numpy as np

from bci3wads.utils import constants
from bci3wads.features.formatter import Formatter


def test_formatter(scores):
    characters = np.array([
        Formatter(s).get_char() for s in scores
    ]).reshape(constants.CHARACTERS.shape)

    np.testing.assert_array_equal(characters, constants.CHARACTERS)
