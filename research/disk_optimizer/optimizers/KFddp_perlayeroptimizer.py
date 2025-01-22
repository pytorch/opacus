# Copyright (c) Xinwei Zhang
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

from functools import partial
from typing import Callable, List, Optional

import torch
from opacus.optimizers.ddp_perlayeroptimizer import _clip_and_accumulate_parameter
from opacus.optimizers.optimizer import _generate_noise
from torch import nn
from torch.optim import Optimizer

from .KFddpoptimizer import KF_DistributedDPOptimizer
from .KFoptimizer import KF_DPOptimizer
from .KFperlayeroptimizer import KF_DPPerLayerOptimizer


class KF_SimpleDistributedPerLayerOptimizer(
    KF_DPPerLayerOptimizer, KF_DistributedDPOptimizer
):
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
        kappa: float = 0.7,
        gamma: float = 0.5,
        **kwargs,
    ):
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            kappa=kappa,
            gamma=gamma,
            **kwargs,
        )


class KF_DistributedPerLayerOptimizer(KF_DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    per layer clipping strategy and is compatible with distributed data parallel
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: List[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        kappa: float = 0.7,
        gamma: float = 0.5,
        **kwargs,
    ):
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.max_grad_norms = max_grad_norm
        max_grad_norm = torch.norm(torch.Tensor(self.max_grad_norms), p=2).item()
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            kappa=kappa,
            gamma=gamma,
            **kwargs,
        )
        self._register_hooks()

    def _add_noise_parameter(self, p: nn.Parameter):
        """
        The reason why we need self is because of generator for secure_mode
        """
        noise = _generate_noise(
            std=self.noise_multiplier * self.max_grad_norm,
            reference=p.summed_grad,
            generator=None,
            secure_mode=self.secure_mode,
        )
        p.grad = p.summed_grad + noise

    @property
    def accumulated_iterations(self) -> int:
        return max([p.accumulated_iterations for p in self.params])

    def _scale_grad_parameter(self, p: nn.Parameter):
        if not hasattr(p, "accumulated_iterations"):
            p.accumulated_iterations = 0
        p.accumulated_iterations += 1
        if self.loss_reduction == "mean":
            p.grad /= (
                self.expected_batch_size * p.accumulated_iterations * self.world_size
            )

    def clip_and_accumulate(self):
        raise NotImplementedError(
            "Clip and accumulate is added per layer in DPDDP Per Layer."
        )

    def add_noise(self):
        raise NotImplementedError("Noise is added per layer in DPDDP Per Layer.")

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        if self.step_hook:
            self.step_hook(self)

        for p in self.params:
            p.accumulated_iterations = 0

        self._is_last_step_skipped = False
        return True

    def _ddp_per_layer_hook(
        self, p: nn.Parameter, max_grad_norm: float, _: torch.Tensor
    ):
        _clip_and_accumulate_parameter(p, max_grad_norm)
        # Equivalent ot _check_skip_next_step but without popping because it has to be done for every parameter p
        if self._check_skip_next_step(pop_next=False):
            return

        if self.rank == 0:
            self._add_noise_parameter(p)
        else:
            p.grad = p.summed_grad
        self._scale_grad_parameter(p)

        return p.grad

    def _register_hooks(self):
        for p, max_grad_norm in zip(self.params, self.max_grad_norms):
            if not p.requires_grad:
                continue

            if not hasattr(p, "ddp_hooks"):
                p.ddp_hooks = []

            p.ddp_hooks.append(
                p.register_hook(partial(self._ddp_per_layer_hook, p, max_grad_norm))
            )
