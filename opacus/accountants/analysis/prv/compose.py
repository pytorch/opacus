from typing import List

import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import convolve

from .prvs import DiscretePRV


def _compose_fourier(dprv: DiscretePRV, num_self_composition: int) -> DiscretePRV:
    if len(dprv) % 2 != 0:
        raise ValueError("Can only compose evenly sized discrete PRVs")

    composed_pmf = irfft(rfft(dprv.pmf) ** num_self_composition)

    m = num_self_composition - 1
    if num_self_composition % 2 == 0:
        m += len(composed_pmf) // 2
    composed_pmf = np.roll(composed_pmf, m)

    domain = dprv.domain.shift_right(dprv.domain.shifts * (num_self_composition - 1))

    return DiscretePRV(pmf=composed_pmf, domain=domain)


def _compose_two(dprv_left: DiscretePRV, dprv_right: DiscretePRV) -> DiscretePRV:
    pmf = convolve(dprv_left.pmf, dprv_right.pmf, mode="same")
    domain = dprv_left.domain.shift_right(dprv_right.domain.shifts)
    return DiscretePRV(pmf=pmf, domain=domain)


def _compose_convolution_tree(dprvs: List[DiscretePRV]) -> DiscretePRV:
    # repeatedly convolve neighbouring PRVs until we only have one left
    while len(dprvs) > 1:
        dprvs_conv = []
        if len(dprvs) % 2 == 1:
            dprvs_conv.append(dprvs.pop())

        for dprv_left, dprv_right in zip(dprvs[:-1:2], dprvs[1::2]):
            dprvs_conv.append(_compose_two(dprv_left, dprv_right))

        dprvs = dprvs_conv
    return dprvs[0]


def compose_heterogeneous(
    dprvs: List[DiscretePRV], num_self_compositions: List[int]
) -> DiscretePRV:
    r"""
    Compose a heterogenous list of PRVs with multiplicity. We use FFT to compose
    identical PRVs with themselves first, then pairwise convolve the remaining PRVs.

    This is the approach taken in https://github.com/microsoft/prv_accountant
    """
    if len(dprvs) != len(num_self_compositions):
        raise ValueError("dprvs and num_self_compositions must have the same length")

    dprvs = [
        _compose_fourier(dprv, num_self_composition)
        for dprv, num_self_composition in zip(dprvs, num_self_compositions)
    ]
    return _compose_convolution_tree(dprvs)
