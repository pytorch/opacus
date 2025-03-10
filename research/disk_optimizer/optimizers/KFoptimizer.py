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
from opacus.optimizers.optimizer import DPOptimizer
from torch.optim import Optimizer
from torch.optim.optimizer import required


logger = logging.getLogger(__name__)
logger.disabled = True


class KF_DPOptimizer(DPOptimizer):
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
        kappa=0.7,
        gamma=0.5,
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
        self.kappa = kappa
        self.gamma = gamma

    @DPOptimizer.grad_samples.setter
    def grad_samples(self, value):
        """
        Set the per sample gradient tensors to zero
        """
        if value is not None:
            for p, v in zip(self.params, value):
                p.grad_sample = v
        else:
            for p in self.params:
                if hasattr(p, "grad_sample"):
                    p.grad_sample = None

    def _compute_one_closure(self, closure=required):
        loss = None
        has_kf_d_t = True
        for p in self.params:
            state = self.state[p]
            if "kf_d_t" not in state:
                has_kf_d_t = False
                continue
            # perturb
            p.data.add_(state["kf_d_t"], alpha=self.gamma)
        with torch.enable_grad():
            loss = closure()
        if has_kf_d_t:
            for p in self.params:
                state = self.state[p]
                # perturb back
                p.data.add_(state["kf_d_t"], alpha=-self.gamma)
        return loss

    def _compute_two_closure(self, closure=required):
        loss = None
        has_kf_d_t = True
        with torch.enable_grad():
            loss = closure()
        for p in self.params:
            state = self.state[p]
            if "kf_d_t" not in state:
                has_kf_d_t = False
                continue
            # perturb
            p.data.add_(state["kf_d_t"], alpha=self.gamma)
        # store first set of gradient
        if has_kf_d_t:
            if self.grad_samples is not None and len(self.grad_samples) != 0:
                self.past_grad_samples = self.grad_samples
                for grad in self.past_grad_samples:
                    grad.mul_(1.0 - self.scaling_factor)
                self.grad_samples = None
            with torch.enable_grad():
                loss = closure()
            for p in self.params:
                state = self.state[p]
                # perturb back
                p.data.add_(state["kf_d_t"], alpha=-self.gamma)
            if self.grad_samples is not None and len(self.grad_samples) != 0:
                for grad, past_grad in zip(self.grad_samples, self.past_grad_samples):
                    grad.mul_(self.scaling_factor).add_(past_grad)
                self.past_grad_samples = None
        return loss

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
