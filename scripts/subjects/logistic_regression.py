import joblib
import pathlib
import numpy as np
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection

from bci3wads.features.estimator import Estimator


seed = 42  # For reproducible results

subject_name = 'Subject_A_Train'
channels_tag = 'channels_11'
model_type = 'logit'
average_signals = True
train_on_all = True
trained_models_path = pathlib.Path.cwd() / 'models' / 'trained'

subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
    subject_name / channels_tag

estimator = Estimator(subject_path)


def create_y_train_epoch(epoch, average_signals=False):
    if average_signals:
        y = np.zeros(epoch.signals.shape[0], dtype=bool)
    else:
        y = np.zeros(epoch.signals.shape[:2], dtype=bool)

    y[epoch.target_char_codes] = True

    return y.ravel()


if average_signals:
    signals_epochs = [epoch.average_signals() for epoch in estimator.epochs]
else:
    signals_epochs = [epoch.signals for epoch in estimator.epochs]

y_train_epochs = [create_y_train_epoch(epoch, average_signals)
                  for epoch in estimator.epochs]

X_train = np.concatenate(signals_epochs)
# `X_train.shape[-1]` corresponds to data dimension.
X_train = X_train.reshape(-1, X_train.shape[-1])

y_train = np.concatenate(y_train_epochs)

if not train_on_all:
    # Stratified shuffle split strategy
    split = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=seed)

    *_, (train_inds, test_inds) = split.split(X_train, y_train)

    X_train, X_test = X_train[train_inds], X_train[test_inds]
    y_train, y_test = y_train[train_inds], y_train[test_inds]

    # Friendly train test split strategy
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=seed)

# Scale the data (important for logistic regression)
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)

# if not train_on_all:
#     X_test = scaler.transform(X_test)

p0 = estimator.get_mismatch_proba()
priors = [p0, 1 - p0]

model = sklearn.linear_model.LogisticRegression(max_iter=1000,
                                                class_weight=None,
                                                random_state=seed)

model.fit(X_train, y_train)

if trained_models_path is not None:
    trained_models_path.mkdir(parents=True, exist_ok=True)
    filename = '_'.join([model_type, subject_name, channels_tag]) + '.pickle'
    joblib.dump(model, trained_models_path.joinpath(filename))

y_train_preds = model.predict(X_train)
print('Training accuracy:', np.sum(y_train == y_train_preds) / len(y_train))

print('-' * 79)

if not train_on_all:
    y_test_preds = model.predict(X_test)
    print('Test Accuracy:', np.sum(y_test == y_test_preds) / len(y_test))

# for (signals_epoch, y_train_epoch) in zip(signals_epochs, y_train_epochs):
#     X = signals_epoch.reshape(-1, signals_epoch.shape[-1])

#     y_preds = model.predict(X)
#     print('Training Accuracy:', np.sum(y_train_epoch == y_preds) / len(y_preds))
