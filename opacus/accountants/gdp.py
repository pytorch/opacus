from .accountant import IAccountant
from .analysis import gdp as privacy_analysis


class GaussianAccountant(IAccountant):
    def __init__(self):
        self.steps = []

    def step(self, *, noise_multiplier: float, sample_rate: float):
        if len(self.steps) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.steps.pop()
            if (
                last_noise_multiplier != noise_multiplier
                or last_sample_rate != sample_rate
            ):
                raise ValueError(
                    "Noise multiplier and sample rate have to stay constant in GaussianAccountant."
                )
            else:
                self.steps = [(last_noise_multiplier, last_sample_rate, num_steps + 1)]

        else:
            self.steps = [(noise_multiplier, sample_rate, 1)]

    def get_epsilon(self, delta: float, poisson: bool = True) -> float:
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            poisson: ``True`` is input batches was sampled via Poisson sampling,
                ``False`` otherwise
        """

        compute_eps = (
            privacy_analysis.compute_eps_poisson
            if poisson
            else privacy_analysis.compute_eps_uniform
        )
        noise_multiplier, sample_rate, num_steps = self.steps.pop()
        return compute_eps(
            steps=num_steps,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=delta,
        )

    def __len__(self):
        return len(self.steps)

    @classmethod
    def mechanism(cls) -> str:
        return "gpd"
