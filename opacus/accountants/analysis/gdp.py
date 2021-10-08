#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
*Based on Google's TF Privacy:* https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/analysis/rdp_accountant.py.
*Here, we update this code to Python 3, and optimize dependencies.*

Implements privacy accounting for Gaussian Differential Privacy.
Applies the Dual and Central Limit Theorem (CLT) to estimate privacy budget of
an iterated subsampled Gaussian Mechanism (by either uniform or Poisson
subsampling).
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm


def compute_mu_uniform(steps, noise_multi, sample_rate):
    """Compute mu from uniform subsampling."""

    c = sample_rate * np.sqrt(steps)
    return (
        np.sqrt(2)
        * c
        * np.sqrt(
            np.exp(noise_multi ** (-2)) * norm.cdf(1.5 / noise_multi)
            + 3 * norm.cdf(-0.5 / noise_multi)
            - 2
        )
    )


def compute_mu_poisson(steps, noise_multi, sample_rate):
    """Compute mu from Poisson subsampling."""

    return np.sqrt(np.exp(noise_multi ** (-2)) - 1) * np.sqrt(steps) * sample_rate


def delta_eps_mu(eps, mu):
    """Compute dual between mu-GDP and (epsilon, delta)-DP."""
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta):
    """Compute epsilon from mu given delta via inverse dual."""

    def f(x):
        """Reversely solve dual by matching delta."""
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method="brentq").root


def compute_eps_uniform(steps, noise_multi, sample_rate, delta):
    """Compute epsilon given delta from inverse dual of uniform subsampling."""

    return eps_from_mu(compute_mu_uniform(steps, noise_multi, sample_rate), delta)


def compute_eps_poisson(steps, noise_multi, sample_rate, delta):
    """Compute epsilon given delta from inverse dual of Poisson subsampling."""

    return eps_from_mu(compute_mu_poisson(steps, noise_multi, sample_rate), delta)
