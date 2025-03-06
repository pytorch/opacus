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

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .optimizer import _generate_noise
from .optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
from torch.distributed._tensor.experimental import implicit_replication


logger = logging.getLogger(__name__)
logger.disabled = True


class FSDPDPOptimizerGhostClipping(DPOptimizerFastGradientClipping):
    """
    ``torch.optim.Optimizer`` wrapper to implement Fast Gradient and Ghost Clipping -- modifies DPOptimizer
    to only add noise to the average gradient, without clipping.

    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimizerFastGradientClipping`` assumes that parameters over which it performs optimization belong
    to GradSampleModuleFastGradientClipping and therefore have the ``grad_sample`` attribute.

    On a high level ``DPOptimizerFastGradientClipping``'s step looks like this:
    1) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    max grad norm limit (``std = noise_multiplier * max_grad_norm``).
    2) Call underlying optimizer to perform optimization step

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimizer = FSDPDPOptimizerFastGradientClipping(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     expected_batch_size=4,
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        """

        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for calculating the standard devition of noise added
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """

        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            expected_batch_size=expected_batch_size,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def accumulate(self):
        """
        Redefines a parent class' function to not do anything
        Gradient accumulation is not supported for this optimizer.
        """
        pass

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad``. ``p.grad_sample`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """

        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample and p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )

        self.original_optimizer.zero_grad(set_to_none)

    def add_noise(self):
        """
        Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``
        """

        for p in self.params:
            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            noise.div_(self.world_size)
            p.grad.add_(noise.view_as(p))

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        # The corner case when the optimizer has no trainable parameters.
        # Essentially the DPOptimizer act as a normal optimizer
        with implicit_replication():
            self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)
        return True
