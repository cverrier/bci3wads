import pathlib
import numpy as np
import sklearn.discriminant_analysis

from bci3wads.utils import data
from bci3wads.features.epoch import Epoch
from bci3wads.features.estimator import Estimator


# TODO:
# (1) Gather all signals from a single subject together,
# (2) Train/Test split the dataset using stratified k-fold strategy,
# (3) Train a model,
# (4) Test the model on the test set created in step (1).


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

p0 = estimator.get_mismatch_proba()
priors = [p0, 1 - p0]

model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
    priors=priors
)
model.fit(X_train, y_train)

y_train_preds = model.predict(X_train)
print('Training accuracy:', np.sum(y_train == y_train_preds) / len(y_train))

print('-' * 79)

for (signals_epoch, y_train_epoch) in zip(signals_epochs, y_train_epochs):
    X = signals_epoch.reshape(-1, signals_epoch.shape[-1])

    y_preds = model.predict(X)
    print('Training Accuracy:', np.sum(y_train_epoch == y_preds) / len(y_preds))

    # for epoch_path in sorted(list(subject_path.glob('epoch_*.pickle')),
    #                          key=lambda p: int(p.stem.split('_')[-1])):
    #     epoch = Epoch(data.load_pickle(epoch_path))

    #     signals = epoch.signals
    #     X_train = signals.reshape(-1, signals.shape[-1])

    #     y_train = np.zeros(signals.shape[:2], dtype=bool)
    #     y_train[epoch.target_char_codes] = True
    #     y_train = y_train.ravel()

    #     # TODO: may be better to do it another way because Estimator class
    #     # is loading all epochs.
    #     estimator = Estimator(subject_path)
    #     p0 = estimator.get_mismatch_proba()
    #     priors = [p0, 1 - p0]

    #     model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
    #         priors=priors
    #     )

    #     model.fit(X_train, y_train)

    #     y_train_preds = model.predict(X_train)
    #     print('Training accuracy:', np.sum(y_train == y_train_preds) / len(y_train))
