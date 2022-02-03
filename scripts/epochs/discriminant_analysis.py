import pathlib
import numpy as np
import sklearn.discriminant_analysis

from bci3wads.utils import data
from bci3wads.features.epoch import Epoch
from bci3wads.features.estimator import Estimator


subject_name = 'Subject_A_Train'
channels_tag = 'channels_11'
model_type = 'lda'

subject_path = pathlib.Path.cwd() / 'data' / 'processed' / \
    subject_name / channels_tag

for epoch_path in sorted(list(subject_path.glob('epoch_*.pickle')),
                         key=lambda p: int(p.stem.split('_')[-1])):
    epoch = Epoch(data.load_pickle(epoch_path))

    signals = epoch.signals
    X_train = signals.reshape(-1, signals.shape[-1])

    y_train = np.zeros(signals.shape[:2], dtype=bool)
    y_train[epoch.target_char_codes] = True
    y_train = y_train.ravel()

    estimator = Estimator(subject_path)
    p0 = estimator.get_mismatch_proba()
    priors = [p0, 1 - p0]

    model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
        priors=priors
    )

    model.fit(X_train, y_train)

    y_train_preds = model.predict(X_train)
    print('Training accuracy:', np.sum(y_train == y_train_preds) / len(y_train))
