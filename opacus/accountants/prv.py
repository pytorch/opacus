# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np

from .accountant import IAccountant
from .analysis.prv import (
    Domain,
    PoissonSubsampledGaussianPRV,
    TruncatedPrivacyRandomVariable,
    compose_heterogeneous,
    compute_safe_domain_size,
    discretize,
)


class PRVAccountant(IAccountant):
    def __init__(self):
        super().__init__()

    def step(self, *, noise_multiplier: float, sample_rate: float):
        if len(self.history) >= 1:
            (last_noise_multiplier, last_sample_rate, num_steps) = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_epsilon(
        self, delta: float, *, eps_error: float = 0.1, delta_error: float = 1e-6
    ) -> float:
        dprv = self._get_dprv(eps_error=eps_error, delta_error=delta_error)
        _, _, eps_upper = dprv.compute_epsilon(delta, delta_error, eps_error)
        return eps_upper

    def _get_dprv(self, eps_error, delta_error):
        # convert history to privacy loss random variables (prvs)
        prvs = [
            PoissonSubsampledGaussianPRV(sample_rate, noise_multiplier)
            for noise_multiplier, sample_rate, _ in self.history
        ]
        num_self_compositions = [steps for _, _, steps in self.history]
        domain = self._get_domain(
            prvs=prvs,
            num_self_compositions=num_self_compositions,
            eps_error=eps_error,
            delta_error=delta_error,
        )
        tprvs = [
            TruncatedPrivacyRandomVariable(prv, domain.t_min, domain.t_max)
            for prv in prvs
        ]
        # discretize and convolve prvs
        dprvs = [discretize(tprv, domain) for tprv in tprvs]
        return compose_heterogeneous(
            dprvs=dprvs, num_self_compositions=num_self_compositions
        )

    def _get_domain(
        self,
        prvs: List[PoissonSubsampledGaussianPRV],
        num_self_compositions: List[int],
        eps_error: float,
        delta_error: float,
    ) -> Domain:
        total_self_compositions = sum(num_self_compositions)

        L = compute_safe_domain_size(
            prvs=prvs,
            max_self_compositions=num_self_compositions,
            eps_error=eps_error,
            delta_error=delta_error,
        )

        mesh_size = eps_error / np.sqrt(
            total_self_compositions * np.log(12 / delta_error) / 2
        )

        return Domain.create_aligned(-L, L, mesh_size)

    @classmethod
    def mechanism(cls) -> str:
        return "prv"

    def __len__(self):
        return len(self.history)
