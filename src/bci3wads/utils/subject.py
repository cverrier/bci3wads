from . import data
from ..features.epoch import Epoch


class Subject:
    def __init__(self, epochs_path):
        epochs = []
        for epoch_path in sorted(list(epochs_path.glob('epoch_*.pickle')),
                                 key=lambda p: int(p.stem.split('_')[-1])):
            epoch = Epoch(data.load_pickle(epoch_path))
            epochs.append(epoch)

        self.epochs = epochs
