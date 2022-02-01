import pathlib
import numpy as np
import matplotlib.pyplot as plt

from bci3wads.features.estimator import Estimator
from bci3wads.features.cov_estimator import CovEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from bci3wads.generators.gaussian_generator import GaussianGenerator

params_type = 'p300'  # 'simulated'

if params_type == 'simulated':
    means = [np.array([1, 2]), np.array([20, 14])]
    covs = [np.array([[1.3, 0], [0, 7.1]]), np.array([[1, 0], [0, 4]])]
    p0 = 0.8
elif params_type == 'p300':
    subject_name = 'Subject_A_Train'
    channels_tag = 'channels_11'

    subject_path = pathlib.Path.cwd() / 'data' / 'processed' / subject_name / \
        channels_tag

    estimator = Estimator(subject_path)
    means = [
        estimator.estimate_target_signal(method=method)
        for method in ['mismatch_avg', 'match_avg']
    ]

    covs = [
        estimator.estimate_noise_cov() for _ in range(len(means))
    ]

    p0 = estimator.get_mismatch_proba()
    p1 = 1 - p0

else:
    raise NotImplementedError()

weights = [p0, 1 - p0]

params = list(zip(means, covs, weights))

generator = GaussianGenerator(*params)
X_train, y_train = generator.mixture(n_obs=50)
X_test, y_test = generator.mixture(n_obs=500)

cov_estimator = CovEstimator()
model = LinearDiscriminantAnalysis(solver='eigen', priors=generator.weights,
                                   covariance_estimator=cov_estimator)

model.fit(X_train, y_train)

training_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print('Training accuracy:',
      np.sum(training_predictions == y_train) / len(y_train))
print('Test accuracy:', np.sum(test_predictions == y_test) / len(y_test))

plt.scatter(*X_train.T, c=y_train)
plt.show()
