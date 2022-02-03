import sys
import pathlib
import numpy as np
import sklearn.model_selection
import sklearn.discriminant_analysis

from bci3wads.utils import data
from bci3wads.features.epoch import Epoch
from bci3wads.features.estimator import Estimator


# TODO:
# (1) Gather all signals from a single subject together,
# (2) Train/Test split the dataset using stratified shuffle split strategy,
# (3) Train a model,
# (4) Test the model on the test set created in step (1).

seed = 42  # For reproducible results

subject_name = 'Subject_A_Train'
channels_tag = 'channels_11'
model_type = 'lda'

subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
    subject_name / channels_tag

estimator = Estimator(subject_path)


def create_y_train_epoch(epoch):
    y = np.zeros(epoch.signals.shape[:2], dtype=bool)
    y[epoch.target_char_codes] = True

    return y.ravel()


signals_epochs = [epoch.signals for epoch in estimator.epochs]
y_train_epochs = [create_y_train_epoch(epoch) for epoch in estimator.epochs]

X_train = np.concatenate(signals_epochs)
# `X_train.shape[-1]` corresponds to data dimension.
X_train = X_train.reshape(-1, X_train.shape[-1])

y_train = np.concatenate(y_train_epochs)

# Stratified shuffle split strategy
split = sklearn.model_selection.StratifiedShuffleSplit(
    n_splits=1, test_size=0.2, random_state=seed)

*_, (train_inds, test_inds) = split.split(X_train, y_train)

X_train, X_test = X_train[train_inds], X_train[test_inds]
y_train, y_test = y_train[train_inds], y_train[test_inds]

# Friendly train test split strategy
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
#     X_train, y_train, test_size=0.2, random_state=seed)

p0 = estimator.get_mismatch_proba()
priors = [p0, 1 - p0]

model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
    priors=priors
)
model.fit(X_train, y_train)

y_train_preds = model.predict(X_train)
print('Training accuracy:', np.sum(y_train == y_train_preds) / len(y_train))

print('-' * 79)

y_test_preds = model.predict(X_test)
print('Test Accuracy:', np.sum(y_test == y_test_preds) / len(y_test))

# for (signals_epoch, y_train_epoch) in zip(signals_epochs, y_train_epochs):
#     X = signals_epoch.reshape(-1, signals_epoch.shape[-1])

#     y_preds = model.predict(X)
#     print('Training Accuracy:', np.sum(y_train_epoch == y_preds) / len(y_preds))
