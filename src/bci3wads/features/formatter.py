import numpy as np

from bci3wads.utils import constants


class Formatter:
    def __init__(self, predictions):
        self.predictions = predictions

    def get_coords(self):
        best_rows_scores = self.predictions[6:]
        best_cols_scores = self.predictions[:6]

        best_row_coord = np.argmax(best_rows_scores)
        best_col_coord = np.argmax(best_cols_scores)

        return [best_row_coord, best_col_coord]

    def get_char(self):
        row_coord, col_coord = self.get_coords()
        return constants.CHARACTERS[row_coord, col_coord]
