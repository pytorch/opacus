from typing import Callable

from .optimizer import DPOptimizer


class _NoiseScheduler(object):
    def __init__(self, optimizer: DPOptimizer, last_epoch=-1):
        # Attach optimizer
        if not isinstance(optimizer, DPOptimizer):
            raise TypeError("{} is not a DPOptimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        self.step()

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_noise_multiplier(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        noise_multiplier = self.get_noise_multiplier()
        self.optimizer.noise_multiplier = noise_multiplier


class ExponentialNoise(_NoiseScheduler):
    def __init__(self, optimizer: DPOptimizer, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_noise_multiplier(self):
        if self.last_epoch == 0:
            return self.optimizer.noise_multiplier
        else:
            return self.optimizer.noise_multiplier * self.gamma


class LambdaNoise(_NoiseScheduler):
    def __init__(
        self,
        optimizer: DPOptimizer,
        noise_lambda: Callable[[int], float],
        last_epoch: int = -1,
    ):
        self.noise_lambda = noise_lambda
        self.base_noise_multiplier = optimizer.noise_multiplier
        super().__init__(optimizer, last_epoch)

    def get_noise_multiplier(self):
        return self.base_noise_multiplier * self.noise_lambda(self.last_epoch)


class StepNoise(_NoiseScheduler):
    def __init__(
        self, optimizer: DPOptimizer, step_size: int, gamma: float, last_epoch: int = -1
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_noise_multiplier(self):
        # Only change noise_multiplier when at a 'step'
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.optimizer.noise_multiplier
        else:
            return self.gamma * self.optimizer.noise_multiplier
