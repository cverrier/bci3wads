import pathlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis

from bci3wads.utils import constants
from bci3wads.features.estimator import Estimator
from bci3wads.features.cov_estimator import CovEstimator
from bci3wads.generators.gaussian_generator import GaussianGenerator

params_type = 'p300'  # 'simulated'
model_type = 'lda'

if params_type == 'simulated':
    means = [np.array([1, 2]), np.array([20, 14])]
    covs = [np.array([[1.3, 0], [0, 7.1]]), np.array([[1, 0], [0, 4]])]

    p0 = 0.8
    weights = [p0, 1 - p0]

    n_training_obs = 500

    # To approximately get the same ratio as in the p300 data
    n_test_obs = int(1.17 * n_training_obs)
elif params_type == 'p300':
    subject_name = 'Subject_A_Train'
    channels_tag = 'channels_11'
    subject_path = pathlib.Path.cwd() / 'data' / 'processed' / subject_name / \
        channels_tag

    estimator = Estimator(subject_path)
    means, covs, weights = estimator.estimate_disc_ana_params(model_type)

    n_training_obs = constants.N_CODES * len(estimator.epochs)
    n_test_obs = constants.N_CODES * 100
else:
    raise NotImplementedError()

params = list(zip(means, covs, weights))

generator = GaussianGenerator(*params)
X_train, y_train = generator.mixture(n_obs=n_training_obs)
X_test, y_test = generator.mixture(n_obs=n_test_obs)

# cov_estimator = CovEstimator()
if model_type == 'lda':
    model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
        priors=generator.weights)
elif model_type == 'qda':
    # May be needed to tweak the `tol` parameter of the classifier to
    # avoid a warning about colinear vectors.
    model = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(
        priors=generator.weights)
    pass
else:
    raise NotImplementedError()

model.fit(X_train, y_train)

training_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print('Training accuracy:',
      np.sum(training_predictions == y_train) / len(y_train))
print('Test accuracy:', np.sum(test_predictions == y_test) / len(y_test))

# Only for simulated two-dimensional data
# plt.scatter(*X_train.T, c=y_train)
# plt.show()
