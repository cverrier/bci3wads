import pathlib

RAW_DATA_PATH = pathlib.Path.cwd() / 'data' / 'raw'
INTER_DATA_PATH = pathlib.Path.cwd() / 'data' / 'intermediate'
PROC_DATA_PATH = pathlib.Path.cwd() / 'data' / 'processed'

WINDOW_SIZE = 240
EPOCH_ID = 0
CHANNEL_ID = 11
CHANNEL_IDS = [11]
