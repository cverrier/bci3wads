import numpy as np
import matplotlib.pyplot as plt

from bci3wads.generators.gaussian_generator import GaussianGenerator

means = [np.array([1, 2]), np.array([10, 9])]
covs = [np.array([[1.3, 0], [0, 7.1]]), np.array([[1, 0], [0, 4]])]
weights = [0.2, 0.8]

params = list(zip(means, covs, weights))

generator = GaussianGenerator(*params)
obs, labels = generator.mixture(n_obs=1000)

plt.scatter(*obs.T, c=labels)
plt.show()
