import abc


class IAccountant(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self, *, noise_multiplier: float, sample_rate: float):
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
