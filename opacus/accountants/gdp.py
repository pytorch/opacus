from typing import List, Optional, Tuple, Union
from .analysis import gdp as privacy_analysis
from .accountant import Accountant


class GaussianAccountant(Accountant):
    def __init__(self, noise_multiplier, sample_rate, poisson):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.poisson = poisson
        self.steps = 0

    def step(self, noise_multiplier: float, sample_rate: float):
        if noise_multiplier != self.noise_multiplier or sample_rate != self.sample_rate:
            raise ValueError(
                "Noise multiplier and sample rate have to stay constant in GaussianAccountant."
            )
        self.steps += 1

    def get_privacy_spent(self, delta: float):
        if self.poisson:
            epsilon = privacy_analysis.compute_eps_poisson(
                self.steps, self.noise_multiplier, self.sample_rate, delta
            )
        else:
            epsilon = privacy_analysis.compute_eps_poisson(
                self.steps, self.noise_multiplier, self.sample_rate, delta
            )

        return epsilon
