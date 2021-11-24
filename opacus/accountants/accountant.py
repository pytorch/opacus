import abc


class IAccountant(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self, noise_multiplier: float, sample_rate: float):
        """
        Signal one optimization step

        Args:
            noise_multiplier: Current noise multiplier
            sample_rate: Current sample rate
        """
        pass

    @abc.abstractmethod
    def get_epsilon(self, delta: float, *args, **kwargs) -> float:
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            *args: subclass-specific args
            **kwargs: subclass-specific kwargs
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Number of optimization steps taken so far
        """
        pass

    @classmethod
    @abc.abstractmethod
    def mechanism(cls) -> str:
        """
        Accounting mechanism name
        """
        pass
