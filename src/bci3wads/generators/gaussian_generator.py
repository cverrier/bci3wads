import numpy as np


class GaussianGenerator:
    def __init__(self, *args):
        self.means, self.covs, self.weights = list(zip(*args))

    def mixture(self, n_obs: int, seed=None):
        rng = np.random.default_rng(seed=seed)

        labels = rng.choice(len(self.weights), size=n_obs, p=self.weights)

        obs = np.array([
            rng.multivariate_normal(self.means[i], self.covs[i])
            for i in labels
        ])

        return obs
