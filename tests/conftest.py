"""
    Dummy conftest.py for bci3wads.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import numpy as np

from bci3wads.utils import constants


@pytest.fixture
def scores():
    n = constants.N_CODES // 2
    template_row_scores = np.eye(n)

    row_scores = np.concatenate([
        np.tile(s, (n, 1)) for s in template_row_scores
    ])
    col_scores = np.tile(template_row_scores, (n, 1))

    return np.concatenate((col_scores, row_scores), axis=1)
