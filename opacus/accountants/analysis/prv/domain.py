from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ...rdp import RDPAccountant


@dataclass
class Domain:
    r"""
    Stores relevant information about the domain on which PRVs are discretized, and
    includes a few convenience methods for manipulating it.
    """
    t_min: float
    t_max: float
    size: int
    shifts: float = 0.0

    def __post_init__(self):
        if not isinstance(self.size, int):
            raise TypeError("`size` must be an integer")
        if self.size % 2 != 0:
            raise ValueError("`size` must be even")

    @classmethod
    def create_aligned(cls, t_min: float, t_max: float, dt: float) -> "Domain":
        t_min = np.floor(t_min / dt) * dt
        t_max = np.ceil(t_max / dt) * dt

        size = int(np.round((t_max - t_min) / dt)) + 1

        if size % 2 == 1:
            size += 1
            t_max += dt

        domain = cls(t_min, t_max, size)

        if np.abs(domain.dt - dt) / dt >= 1e-8:
            raise RuntimeError

        return domain

    def shift_right(self, dt: float) -> "Domain":
        return Domain(
            t_min=self.t_min + dt,
            t_max=self.t_max + dt,
            size=self.size,
            shifts=self.shifts + dt,
        )

    @property
    def dt(self):
        return (self.t_max - self.t_min) / (self.size - 1)

    @property
    def ts(self):
        return np.linspace(self.t_min, self.t_max, self.size)

    def __getitem__(self, i: int) -> float:
        return self.t_min + i * self.dt


def compute_safe_domain_size(
    prvs,
    max_self_compositions: Sequence[int],
    eps_error: float,
    delta_error: float,
) -> float:
    """
    Compute safe domain size for the discretization of the PRVs.

    For details about this algorithm, see remark 5.6 in
    https://www.microsoft.com/en-us/research/publication/numerical-composition-of-differential-privacy/
    """
    total_compositions = sum(max_self_compositions)

    rdp_accountant = RDPAccountant()
    for prv, max_self_composition in zip(prvs, max_self_compositions):
        rdp_accountant.history.append(
            (prv.noise_multiplier, prv.sample_rate, max_self_composition)
        )

    L_max = rdp_accountant.get_epsilon(delta_error / 4)

    for prv, max_self_composition in zip(prvs, max_self_compositions):
        rdp_accountant = RDPAccountant()
        rdp_accountant.history = [(prv.noise_multiplier, prv.sample_rate, 1)]
        L_max = max(
            L_max,
            rdp_accountant.get_epsilon(delta=delta_error / (8 * total_compositions)),
        )

    # FIXME: this implementation is adapted from the code accompanying the paper, but
    # disagrees subtly with the theory from remark 5.6. It's not immediately clear this
    # gives the right guarantees in all cases, though it's fine for eps_error < 1 and
    # hence generic cases.
    # cf. https://github.com/microsoft/prv_accountant/discussions/34
    return max(L_max, eps_error) + 3
