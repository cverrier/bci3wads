import pathlib
import numpy as np

from bci3wads.utils import data
from bci3wads.utils import constants
from bci3wads.features.epoch import Epoch
from bci3wads.features.estimator import Estimator


subject_name = 'Subject_A_Train'
channels_tag = 'channels_11'
model_type = 'logit'

subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
    subject_name / channels_tag

predictions_path = constants.PREDICTIONS_PATH / \
    model_type / subject_name / channels_tag

# TODO: No need of estimator, create a more appropriate class... Later.
estimator = Estimator(subject_path)
target_chars = np.array([epoch.target_char for epoch in estimator.epochs])

predicted_chars = np.load(predictions_path.joinpath('predictions.npy'))

print('Accuracy:', np.sum(target_chars == predicted_chars) / len(target_chars))
