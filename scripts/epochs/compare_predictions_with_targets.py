import pathlib
import numpy as np
import matplotlib.pyplot as plt

from bci3wads.utils import data
from bci3wads.utils import constants
from bci3wads.features.epoch import Epoch
from bci3wads.features.estimator import Estimator
from bci3wads.features.formatter import Formatter


subject_name = 'Subject_A_Train'
channels_tag = 'channels_11'
model_name = 'improved_detector'

subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
    subject_name / channels_tag

predictions_path = constants.PREDICTIONS_PATH / \
    model_name / subject_name / channels_tag

epochs = [
    Epoch(data.load_pickle(path))
    for path in sorted(list(subject_path.glob('epoch_*.pickle')),
                       key=lambda p: int(p.stem.split('_')[-1]))
]

target_chars = np.array([epoch.target_char for epoch in epochs])

accuracies = []

for n_trials in range(1, constants.N_TRIALS + 1):
    filename = f'predictions_{n_trials}_trials.npy'
    predicted_chars = np.load(predictions_path.joinpath(filename))

    accuracy = np.sum(target_chars == predicted_chars) / len(target_chars)
    accuracies.append(accuracy)
    # print('Accuracy:', accuracy)

plt.plot(np.arange(1, constants.N_TRIALS + 1), accuracies)
plt.xlabel('Number of trials')
plt.ylabel('Accuracy')
plt.title('Accuracy with Respect to Number of Trials')
plt.show()
