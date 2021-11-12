from typing import List, Optional, Tuple, Union

from .accountant import IAccountant
from .analysis import rdp as privacy_analysis


class RDPAccountant(IAccountant):
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self, alphas=None):
        self.steps = []
        self.alphas = alphas

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
            if self.alphas is None:
                alphas = self.DEFAULT_ALPHAS
            else:
                alphas = self.alphas

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

    def get_epsilon(self, delta: float):
        eps, _ = self.get_privacy_spent(delta)
        return eps


def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    accountant: RDPAccountant = RDPAccountant(),
    sigma_min: Optional[float] = 0.01,
    sigma_max: Optional[float] = 10.0,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        alphas: the list of orders at which to compute RDP
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    eps = float("inf")
    while eps > target_epsilon:
        sigma_max = 2 * sigma_max
        accountant.steps = [(sigma_max, sample_rate, int(epochs / sample_rate))]
        eps = accountant.get_privacy_spent(target_delta)[0]
        if sigma_max > 2000:
            raise ValueError("The privacy budget is too low.")

    while sigma_max - sigma_min > 0.01:
        sigma = (sigma_min + sigma_max) / 2
        accountant.steps = [(sigma, sample_rate, int(epochs / sample_rate))]
        eps = accountant.get_privacy_spent(target_delta)[0]

        if eps < target_epsilon:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma
