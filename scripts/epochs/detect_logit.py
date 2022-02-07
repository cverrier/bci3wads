import joblib
import pathlib
import numpy as np

from bci3wads.utils import constants
from bci3wads.features.estimator import Estimator
from bci3wads.features.formatter import Formatter


subject_name = 'Subject_A_Train'
channels_tag = 'channels_11'
model_type = 'logit'

subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
    subject_name / channels_tag

results_path = constants.PREDICTIONS_PATH / model_type / subject_name / \
    channels_tag
results_path.mkdir(parents=True, exist_ok=True)

models_path = pathlib.Path.cwd() / 'models' / 'trained'
filename = '_'.join([model_type, subject_name, channels_tag]) + '.pickle'

model = joblib.load(models_path.joinpath(filename))
disc_vec = model.coef_.ravel()  # Be careful: a view, not a copy

# TODO: Create a more appropriate class to load epochs, creating an
# Estimator here does not really make sense as we do not estimate
# anything.
estimator = Estimator(subject_path)  # Epochs are now loaded in order

predicted_chars = []
for epoch in estimator.epochs:
    signals = epoch.average_signals()
    scores = signals @ disc_vec

    formatter = Formatter(scores)
    predicted_char = formatter.get_char()
    predicted_chars.append(predicted_char)

np.save(results_path.joinpath('predictions.npy'), predicted_chars)
