import pathlib
import numpy as np

RAW_DATA_PATH = pathlib.Path.cwd() / 'data' / 'raw'
INTER_DATA_PATH = pathlib.Path.cwd() / 'data' / 'intermediate'
PROC_DATA_PATH = pathlib.Path.cwd() / 'data' / 'processed'

WINDOW_SIZE = 240
EPOCH_ID = 0
CHANNEL_ID = 11
CHANNEL_IDS = [11]

CHARACTERS = np.array([
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_']
])
