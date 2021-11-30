from __future__ import annotations

from functools import partial
from typing import List, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from .optimizer import DPOptimizer, _generate_noise


def _clip_and_accumulate_parameter(p: nn.Parameter, max_grad_norm: float):
    per_sample_norms = p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=-1)
    per_sample_clip_factor = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

    grad = torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample)

    if hasattr(p, "summed_grad"):
        p.summed_grad += grad
    else:
        p.summed_grad = grad


class DistributedPerLayerOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    per layer clipping strategy and is compatible with distibured data parallel
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norms: List[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        self.rank = torch.distributed.get_rank()
        self.max_grad_norms = max_grad_norms
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

    def _scale_grad_parameter(self, p: nn.Parameter):
        if not hasattr(p, "accumulated_iterations"):
            p.accumulated_iterations = 0
        p.accumulated_iterations += 1
        if self.loss_reduction == "mean":
            p.grad /= self.expected_batch_size * p.accumulated_iterations

    def clip_and_accumulate(self):
        raise NotImplementedError(
            "Clip and accumulate is added per layer in DPDDP Per Layer."
        )

    def add_noise(self):
        raise NotImplementedError("Noise is added per layer in DPDDP Per Layer.")

    def pre_step(self):
        self.accumulated_iterations = max(
            [p.accumulated_iterations for p in self.params]
        )
        if self.step_hook:
            self.step_hook(self)
        self.accumulated_iterations = 0
        for p in self.params:
            p.accumulated_iterations = 0

    def _ddp_per_layer_hook(
        self, p: nn.Parameter, max_grad_norm: float, _: torch.Tensor
    ):
        _clip_and_accumulate_parameter(p, max_grad_norm)
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
