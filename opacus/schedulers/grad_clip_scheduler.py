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


class _GradClipScheduler:
    """Base class for gradient clipping schedulers. We follow the same API
    as the standard PyTorch LR schedulers, but apply them to Opacus's
    `max_grad_norm` param instead.

    This means it only works when you pass a opacus.DPOptimizer, since that
    will have a `max_grad_norm` attribute.
    """

    def __init__(self, optimizer: DPOptimizer, *, last_epoch=-1):
        """
        Args:
            optimizer (DPOptimizer): The DPOptimizer
            *: Any other positional args (this is an abstract base class)
            last_epoch(int): The index of last epoch. Default: -1.
        """
        if not hasattr(optimizer, "max_grad_norm"):
            raise ValueError(
                "GradClipSchedulers require your optimizer to have a .max_grad_norm attr. "
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

    def get_max_grad_norm(self):
        """Implement your scheduling logic here and return the new value for `max_grad_norm`."""
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        max_grad_norm = self.get_max_grad_norm()
        self.optimizer.max_grad_norm = max_grad_norm


class ExponentialGradClip(_GradClipScheduler):
    """
    Multiplies the max_grad_norm by gamma every epoch (so the gamma factors accumulate).
    This means that:
        - For gamma < 1, max_grad_norm will shrink and you'll clip more
        - For gamma == 1, no effect
        - For gamma > 1, max_grad_norm will expand so you'll clip less

    When last_epoch=-1, sets initial max_grad_norm as max_grad_norm.
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

    def get_max_grad_norm(self):
        if self.last_epoch == 0:
            return self.optimizer.max_grad_norm
        else:
            return self.optimizer.max_grad_norm * self.gamma


class LambdaGradClip(_GradClipScheduler):
    """
    Multiplies your *base* `max_grad_norm` by the output of a `scheduler_function` given
    as input.
    Note: the base max_grad_norm is recorded as the max_grad_norm your optimizer had set at
    the very beginning. This means that the factors from the `scheduler_function` will *not*
    accumulate, unlike in ExponentialGradClip. If you want some exponential-like behavior,
    accumulation logic will have to be added in your `scheduler_function`.

    When last_epoch=-1, sets initial max_grad_norm as max_grad_norm.
    """

    def __init__(
        self,
        optimizer: DPOptimizer,
        *,
        scheduler_function: Callable[[int], float],
        last_epoch: int = -1,
    ):
        """

        Args:
            optimizer: Wrapped optimizer.
            scheduler_function: A function which computes a multiplicative factor given
                an integer epoch
            last_epoch: The index of last epoch. Default: -1.
        """
        self.scheduler_function = scheduler_function
        self.base_max_grad_norm = optimizer.max_grad_norm
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_max_grad_norm(self):
        return self.base_max_grad_norm * self.scheduler_function(self.last_epoch)


class StepGradClip(_GradClipScheduler):
    """
    Multiplies `max_grad_norm` by `gamma` every `step_size` epochs (so the `gamma` factors accumulate).
    This means that:
        - For gamma < 1, max_grad_norm will shrink and you'll clip more
        - For gamma == 1, no effect
        - For gamma > 1, max_grad_norm will expand so you'll clip less

    When last_epoch=-1, sets initial max_grad_norm as max_grad_norm.
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

    def get_max_grad_norm(self):
        # Only change max_grad_norm when at a 'step'
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.optimizer.max_grad_norm
        else:
            return self.gamma * self.optimizer.max_grad_norm
