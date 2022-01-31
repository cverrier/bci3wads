import pathlib
import numpy as np

RAW_DATA_PATH = pathlib.Path.cwd() / 'data' / 'raw'
INTER_DATA_PATH = pathlib.Path.cwd() / 'data' / 'intermediate'
PROC_DATA_PATH = pathlib.Path.cwd() / 'data' / 'processed'

PREDICTIONS_PATH = pathlib.Path.cwd() / 'models' / 'predictions'

N_CODES = 12
N_TRIALS = 15
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

REF_SUBJECT_PATH = PROC_DATA_PATH / 'Subject_A_Train' / 'channels_11'
REF_EPOCH_PATH = REF_SUBJECT_PATH / 'epoch_0.pickle'
