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

from typing import Callable, Dict

from opacus.optimizers import DPOptimizer


class _NoiseScheduler:
    """Base class for noise multiplier schedulers. We follow the same API
    as the standard PyTorch LR schedulers, but apply them to Opacus's noise
    multiplier param instead.

    This means it only works when you pass a opacus.DPOptimizer, since that
    will have a `noise_multiplier` attribute.
    """

    def __init__(self, optimizer: DPOptimizer, *, last_epoch=-1):
        """
        Args:
            optimizer (DPOptimizer): The DPOptimizer
            *: Any other positional args (this is an abstract base class)
            last_epoch(int): The index of last epoch. Default: -1.
        """
        if not hasattr(optimizer, "noise_multiplier"):
            raise ValueError(
                "NoiseSchedulers require your optimizer to have a .noise_multiplier attr. "
                "Are you sure you are using a DPOptimizer? Those have it added for you."
            )
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
        """Implement your scheduling logic here and return the new value for `noise_multiplier`."""
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        noise_multiplier = self.get_noise_multiplier()
        self.optimizer.noise_multiplier = noise_multiplier


class ExponentialNoise(_NoiseScheduler):
    """
    Multiplies the noise_multiplier by gamma every epoch (so the gamma factors accumulate).
    This means that:
        - For gamma < 1, noise_multiplier will shrink
        - For gamma == 1, no effect
        - For gamma > 1, noise_multiplier will expand

    When last_epoch=-1, sets initial noise_multiplier as noise_multiplier.

    """

    def __init__(self, optimizer: DPOptimizer, *, gamma: float, last_epoch: int = -1):
        """

        Args:
            optimizer: Wrapped optimizer
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch. Default: -1.
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
    Multiplies your *base* `noise_multiplier` by the output of a `scheduler_function` given
    as input.
    Note: the base noise_multiplier is recorded as the noise_multiplier your optimizer
    had set at the very beginning. This means that the factors from the `scheduler_function`
    will *not* accumulate, unlike in ExponentialGradClip.
    If you want some exponential-like behavior, accumulation logic will have to be
    added in your `scheduler_function`.

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
    Multiplies `noise_multiplier` by `gamma` every `step_size` epochs (so the `gamma` factors accumulate).
    This means that:
        - For gamma < 1, noise_multiplier will shrink
        - For gamma == 1, no effect
        - For gamma > 1, noise_multiplier will expand

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
