from typing import List, Optional, Tuple, Union

from .accountant import IAccountant
from .analysis import rdp as privacy_analysis


class RDPAccountant(IAccountant):
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self):
        self.steps = []

    def step(self, noise_multiplier: float, sample_rate: float):
        if len(self.steps) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.steps.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.steps.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.steps.append((last_noise_multiplier, last_sample_rate, num_steps))
                self.steps.append((noise_multiplier, sample_rate, 1))

        else:
            self.steps.append((noise_multiplier, sample_rate, 1))

    def get_privacy_spent(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ) -> Tuple[float, float]:
        if alphas is None:
            alphas = self.DEFAULT_ALPHAS

        rdp = sum(
            [
                privacy_analysis.compute_rdp(
                    sample_rate, noise_multiplier, num_steps, alphas
                )
                for (noise_multiplier, sample_rate, num_steps) in self.steps
            ]
        )

        eps, best_alpha = privacy_analysis.get_privacy_spent(alphas, rdp, delta)

        return float(eps), float(best_alpha)

    def get_epsilon(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        eps, _ = self.get_privacy_spent(delta, alphas)
        return eps

    def __len__(self):
        return len(self.steps)
