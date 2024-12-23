from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required

from .KFoptimizer import KF_DPOptimizer


logger = logging.getLogger(__name__)
logger.disabled = True


class KF_DPOptimizerFastGradientClipping(KF_DPOptimizer):
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
    ):
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
        )

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
            for p in self.params:
                p.past_grad = p.grad
                p.past_grad.mul_(1.0 - self.scaling_factor)
                p.grad = None
            with torch.enable_grad():
                loss = closure()
            for p in self.params:
                state = self.state[p]
                # perturb back
                p.data.add_(state["kf_d_t"], alpha=-self.gamma)
            for p in self.params:
                p.grad.mul_(self.scaling_factor).add_(p.past_grad)
                p.past_grad = None
        return loss
