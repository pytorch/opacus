from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import integrate
from scipy.special import erfc

from ..rdp import _compute_rdp
from .domain import Domain


SQRT2 = np.sqrt(2)


class PoissonSubsampledGaussianPRV:
    r"""
    A Poisson subsampled Gaussian privacy random variable.

    For details about the formulas for the pdf and cdf, see propositions B1 and B4 in
    https://www.microsoft.com/en-us/research/publication/numerical-composition-of-differential-privacy/
    """

    def __init__(self, sample_rate: float, noise_multiplier: float) -> None:
        self.sample_rate = sample_rate
        self.noise_multiplier = noise_multiplier

    def pdf(self, t):
        q = self.sample_rate
        sigma = self.noise_multiplier

        z = np.log((np.exp(t) + q - 1) / q)

        return np.where(
            t > np.log(1 - q),
            sigma
            * np.exp(-(sigma**2) * z**2 / 2 - 1 / (8 * sigma**2) + 2 * t)
            / (
                SQRT2
                * np.sqrt(np.pi)
                * (np.exp(t) + q - 1)
                * ((np.exp(t) + q - 1) / q) ** 0.5
            ),
            0.0,
        )

    def cdf(self, t):
        q = self.sample_rate
        sigma = self.noise_multiplier

        z = np.log((np.exp(t) + q - 1) / q)

        return np.where(
            t > np.log(1 - q),
            -q * erfc((2 * z * sigma**2 - 1) / (2 * SQRT2 * sigma)) / 2
            - (1 - q) * erfc((2 * z * sigma**2 + 1) / (2 * SQRT2 * sigma)) / 2
            + 1.0,
            0.0,
        )

    def rdp(self, alpha: float) -> float:
        return _compute_rdp(self.sample_rate, self.noise_multiplier, alpha)


# though we have only implemented the PoissonSubsampledGaussianPRV, this truncated prv
# class is generic, and would work with PRVs corresponding to different mechanisms
class TruncatedPrivacyRandomVariable:
    def __init__(
        self, prv: PoissonSubsampledGaussianPRV, t_min: float, t_max: float
    ) -> None:
        self._prv = prv
        self.t_min = t_min
        self.t_max = t_max
        self._remaining_mass = self._prv.cdf(t_max) - self._prv.cdf(t_min)

    def pdf(self, t):
        return np.where(
            t < self.t_min,
            0.0,
            np.where(t < self.t_max, self._prv.pdf(t) / self._remaining_mass, 0.0),
        )

    def cdf(self, t):
        return np.where(
            t < self.t_min,
            0.0,
            np.where(
                t < self.t_max,
                (self._prv.cdf(t) - self._prv.cdf(self.t_min)) / self._remaining_mass,
                1.0,
            ),
        )

    def mean(self) -> float:
        """
        Calculate the mean using numerical integration.
        """
        points = np.concatenate(
            [
                [self.t_min],
                -np.logspace(-5, -1, 5)[::-1],
                np.logspace(-5, -1, 5),
                [self.t_max],
            ]
        )

        mean = 0.0
        for left, right in zip(points[:-1], points[1:]):
            integral, _ = integrate.quad(self.cdf, left, right, limit=500)
            mean += right * self.cdf(right) - left * self.cdf(left) - integral

        return mean


@dataclass
class DiscretePRV:
    pmf: np.ndarray
    domain: Domain

    def __len__(self) -> int:
        if len(self.pmf) != self.domain.size:
            raise ValueError("pmf and domain must have the same length")
        return len(self.pmf)

    def compute_epsilon(
        self, delta: float, delta_error: float, eps_error: float
    ) -> Tuple[float, float, float]:
        if delta <= 0:
            return (float("inf"),) * 3

        if np.finfo(np.longdouble).eps * self.domain.size > delta - delta_error:
            raise ValueError(
                "Floating point errors will dominate for such small values of delta. "
                "Increase delta or reduce domain size."
            )

        t = self.domain.ts
        p = self.pmf
        d1 = np.flip(np.flip(p).cumsum())
        d2 = np.flip(np.flip(p * np.exp(-t)).cumsum())
        ndelta = np.exp(t) * d2 - d1

        def find_epsilon(delta_target):
            i = np.searchsorted(ndelta, -delta_target, side="left")
            if i <= 0:
                raise RuntimeError("Cannot compute epsilon")
            return np.log((d1[i] - delta_target) / d2[i])

        eps_upper = find_epsilon(delta - delta_error) + eps_error
        eps_lower = find_epsilon(delta + delta_error) - eps_error
        eps_estimate = find_epsilon(delta)
        return eps_lower, eps_estimate, eps_upper

    def compute_delta_estimate(self, eps: float) -> float:
        return np.where(
            self.domain.ts >= eps,
            self.pmf * (1.0 - np.exp(eps) * np.exp(-self.domain.ts)),
            0.0,
        ).sum()


def discretize(prv, domain: Domain) -> DiscretePRV:
    tC = domain.ts
    tL = tC - domain.dt / 2
    tR = tC + domain.dt / 2
    discrete_pmf = prv.cdf(tR) - prv.cdf(tL)

    mean_d = np.dot(domain.ts, discrete_pmf)
    mean_c = prv.mean()

    mean_shift = mean_c - mean_d

    if np.abs(mean_shift) >= domain.dt / 2:
        raise RuntimeError("Discrete mean differs significantly from continuous mean.")

    domain_shifted = domain.shift_right(mean_shift)

    return DiscretePRV(pmf=discrete_pmf, domain=domain_shifted)
