import abc
from opacus.optimizers import DPOptimizer
from typing import Callable

class IAccountant(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self, noise_multiplier: float, sample_rate: float):
        pass

    @abc.abstractmethod
    def get_epsilon(self, delta: float, *args, **kwargs) -> float:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @classmethod
    @abc.abstractmethod
    def mechanism(cls) -> str:
        pass

    def get_optimizer_hook_fn(self, sample_rate: float) -> Callable[[DPOptimizer], None]:
        def hook_fn(optim: DPOptimizer):
            # This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in
            # distributed mode, each node samples among a subset of the data)
            self.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )
        return hook_fn
