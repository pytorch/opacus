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

from typing import List, Optional

import torch
from opacus.optimizers.utils import params
from opt_einsum import contract
from torch.optim import Optimizer

from .optimizer import DPOptimizer, _check_processed_flag, _mark_as_processed


class DPPerLayerOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    per layer clipping strategy
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
    ):
        assert len(max_grad_norm) == len(params(optimizer))
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
        )

    def clip_and_accumulate(self):
        for p, max_grad_norm in zip(self.params, self.max_grad_norms):
            _check_processed_flag(p.grad_sample)

            grad_sample = self._get_flat_grad_sample(p)
            per_sample_norms = grad_sample.norm(
                2, dim=tuple(range(1, grad_sample.ndim))
            )
            per_sample_clip_factor = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)
