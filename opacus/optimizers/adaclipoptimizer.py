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
from typing import Optional

import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _get_flat_grad_sample,
    _mark_as_processed,
)


logger = logging.getLogger(__name__)


class AdaClipDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    adaptive clipping strategy
    https://arxiv.org/pdf/1905.03871.pdf
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        target_unclipped_quantile: float,
        clipbound_learning_rate: float,
        max_clipbound: float,
        min_clipbound: float,
        fraction_std: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        assert (
            max_clipbound > min_clipbound
        ), "max_clipbound must be larger than min_clipbound."
        self.target_unclipped_quantile = target_unclipped_quantile
        self.clipbound_learning_rate = clipbound_learning_rate
        self.max_clipbound = max_clipbound
        self.min_clipbound = min_clipbound
        self.fraction_std = fraction_std
        self.noise_multiplier_delta = (
            self.noise_multiplier ** (-2) - (2 * fraction_std) ** (-2)
        ) ** (-1 / 2)
        self.sample_size = 0
        self.unclipped_num = 0

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients, self.sample_size and self.unclipped_num
        """

        if set_to_none is False:
            logger.info(
                "Despite set_to_none is set to False, "
                "opacus will set p.grad_sample and p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )

        for p in self.params:
            p.grad_sample = None

            if not self._is_last_step_skipped:
                p.summed_grad = None

        self.original_optimizer.zero_grad(set_to_none)

        self.sample_size = 0
        self.unclipped_num = 0

    def clip_and_accumulate(self):
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        self.sample_size += len(per_sample_clip_factor)
        self.unclipped_num += (
            len(per_sample_clip_factor) - (per_sample_clip_factor < 1).sum()
        )

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier_delta * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p.grad)

            _mark_as_processed(p.summed_grad)

        noise = _generate_noise(
            std=self.fraction_std,
            reference=self.unclipped_num,
            generator=self.generator,
        )

        self.unclipped_num = float(self.unclipped_num)
        self.unclipped_num += noise

    def scale_grad(self):
        """
        Update clipping bound based on unclipped fraction
        """
        unclipped_frac = self.unclipped_num / self.sample_size
        self.max_grad_norm *= torch.exp(
            -self.clipbound_learning_rate
            * (unclipped_frac - self.target_unclipped_quantile)
        )
        if self.max_grad_norm > self.max_clipbound:
            self.max_grad_norm = self.max_clipbound
        elif self.max_grad_norm < self.min_clipbound:
            self.max_grad_norm = self.min_clipbound

        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations
