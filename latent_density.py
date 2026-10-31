"""Fit a Gaussian mixture to training posterior means for sampling."""

import warnings

import numpy as np
from sklearn.mixture import GaussianMixture


def fit_gaussian_mixture(latents, components, seed):
    latents = np.asarray(latents, dtype=np.float64)
    if not 1 <= components < len(latents):
        raise ValueError("components must be between 1 and the number of latents - 1")
    mixture = GaussianMixture(n_components=components, covariance_type="full", reg_covar=1e-5,
                              max_iter=1000, n_init=3, random_state=seed)
    mixture.fit(latents)
    if not mixture.converged_:
        warnings.warn(f"The {components}-component mixture did not converge")
    return mixture


def sample_gaussian_mixture(mixture, count, temperature=1.0, seed=0):
    """Draw z = m_k + temperature * L_k eps with k ~ the mixture weights."""
    rng = np.random.default_rng(seed)
    weights = mixture.weights_ / mixture.weights_.sum()
    labels = rng.choice(len(weights), size=count, p=weights)
    noise = rng.standard_normal((count, mixture.means_.shape[1]))
    samples = np.empty_like(noise)
    for component, (mean, covariance) in enumerate(zip(mixture.means_, mixture.covariances_)):
        chosen = labels == component
        samples[chosen] = mean + np.einsum(
            "ij,kj->ik", temperature * noise[chosen],
            np.linalg.cholesky(covariance), optimize=False)
    return samples
