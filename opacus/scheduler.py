from typing import Callable, Dict

from .optimizers import DPOptimizer


class _NoiseScheduler(object):
    def __init__(self, optimizer: DPOptimizer, *, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        self.step()

    def state_dict(self) -> Dict:
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.

        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: Dict):
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
    """
    Decays the noise_multiplier by gamma every epoch.
    When last_epoch=-1, sets initial noise_multiplier as noise_multiplier.

    """

    def __init__(self, optimizer: DPOptimizer, *, gamma: float, last_epoch: int = -1):
        """

        Args:
            optimizer: Wrapped optimizer
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch
        """
        self.gamma = gamma
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_noise_multiplier(self):
        if self.last_epoch == 0:
            return self.optimizer.noise_multiplier
        else:
            return self.optimizer.noise_multiplier * self.gamma


class LambdaNoise(_NoiseScheduler):
    """
    Sets the noise_multiplier to the initial noise_multiplier times a given function.
    When last_epoch=-1, sets initial noise_multiplier as noise_multiplier.

    """

    def __init__(
        self,
        optimizer: DPOptimizer,
        *,
        noise_lambda: Callable[[int], float],
        last_epoch: int = -1,
    ):
        """

        Args:
            optimizer: Wrapped optimizer.
            noise_lambda: A function which computes a multiplicative factor given
                an integer epoch
            last_epoch: The index of last epoch. Default: -1.
        """
        self.noise_lambda = noise_lambda
        self.base_noise_multiplier = optimizer.noise_multiplier
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_noise_multiplier(self):
        return self.base_noise_multiplier * self.noise_lambda(self.last_epoch)


class StepNoise(_NoiseScheduler):
    """
    Decays the noise_multiplier by gamma every step_size epochs.
    When last_epoch=-1, sets initial noise_multiplier as noise_multiplier.

    """

    def __init__(
        self,
        optimizer: DPOptimizer,
        *,
        step_size: int,
        gamma: float,
        last_epoch: int = -1,
    ):
        """

        Args:
            optimizer: Wrapped optimizer.
            step_size: Period of learning rate decay.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_noise_multiplier(self):
        # Only change noise_multiplier when at a 'step'
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.optimizer.noise_multiplier
        else:
            return self.gamma * self.optimizer.noise_multiplier
