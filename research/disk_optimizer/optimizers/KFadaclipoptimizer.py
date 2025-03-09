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

import logging
import math
from typing import Optional

import torch
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from torch.optim import Optimizer
from torch.optim.optimizer import required

from .KFoptimizer import KF_DPOptimizer


logger = logging.getLogger(__name__)


class KF_AdaClipDPOptimizer(AdaClipDPOptimizer, KF_DPOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        target_unclipped_quantile: float,
        clipbound_learning_rate: float,
        max_clipbound: float,
        min_clipbound: float,
        unclipped_num_std: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        kappa: float = 0.7,
        gamma: float = 0.5,
        **kwargs,
    ):
        if gamma == 0 or abs(gamma - (1 - kappa) / kappa) < 1e-3:
            gamma = (1 - kappa) / kappa
            self.kf_compute_grad_at_original = False
        else:
            self.scaling_factor = (1 - kappa) / (
                gamma * kappa
            )  # (gamma*kappa+kappa-1)/(1-kappa)
            self.kf_compute_grad_at_original = True
            c = (1 - kappa) / (gamma * kappa)
            norm_factor = math.sqrt(c**2 + (1 - c) ** 2)
            noise_multiplier = noise_multiplier / norm_factor
        super(AdaClipDPOptimizer).__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            target_unclipped_quantile=target_unclipped_quantile,
            clipbound_learning_rate=clipbound_learning_rate,
            max_clipbound=max_clipbound,
            min_clipbound=min_clipbound,
            unclipped_num_std=unclipped_num_std,
            **kwargs,
        )
        self.kappa = kappa
        self.gamma = gamma

    def step(self, closure=required) -> Optional[float]:
        if self.kf_compute_grad_at_original:
            loss = self._compute_two_closure(closure)
        else:
            loss = self._compute_one_closure(closure)

        if self.pre_step():
            tmp_states = []
            first_step = False
            for p in self.params:
                grad = p.grad
                state = self.state[p]
                if "kf_d_t" not in state:
                    state = dict()
                    first_step = True
                    state["kf_d_t"] = torch.zeros_like(p.data).to(p.data)
                    state["kf_m_t"] = grad.clone().to(p.data)
                state["kf_m_t"].lerp_(grad, weight=1 - self.kappa)
                p.grad = state["kf_m_t"].clone().to(p.data)
                state["kf_d_t"] = -p.data.clone().to(p.data)
                if first_step:
                    tmp_states.append(state)
            self.original_optimizer.step()
            for p in self.params:
                if first_step:
                    tmp_state = tmp_states.pop(0)
                    self.state[p]["kf_d_t"] = tmp_state["kf_d_t"]
                    self.state[p]["kf_m_t"] = tmp_state["kf_m_t"]
                    del tmp_state
                self.state[p]["kf_d_t"].add_(p.data, alpha=1.0)
        return loss
