from __future__ import annotations

from typing import Callable, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer


def _generate_noise(std: float, reference: torch.Tensor) -> torch.Tensor:
    if std > 0:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
        )
    return torch.zeros(reference.shape, device=reference.device)


class DPOptimizer(Optimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
    ):
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")

        if loss_reduction == "mean" and expected_batch_size is None:
            raise ValueError(
                "You must provide expected batch size of the loss reduction is mean"
            )

        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None

    @property
    def params(self) -> List[nn.Parameter]:
        ret = []
        for param_group in self.optimizer.param_groups:
            ret += [p for p in param_group["params"] if p.requires_grad]
        return ret

    @property
    def grad_samples(self) -> List[torch.Tensor]:
        ret = []
        for p in self.params:
            if not hasattr(p, "grad_sample"):
                raise ValueError(
                    "Per sample gradient not found. Are you using GradSampleModule?"
                )

            ret.append(p.grad_sample)
        return ret

    @property
    def accumulated_iterations(self):
        return len(self.params[0].clipped_grads)

    def attach_step_hook(self, fn: Callable[[DPOptimizer], None]):
        self.step_hook = fn

    def accumulate(self):
        for p in self.params:
            p.summed_grad = sum(p.clipped_grads)

    def add_noise(self):
        for p in self.params:
            noise = _generate_noise(
                self.noise_multiplier * self.max_grad_norm, p.summed_grad
            )
            p.grad = p.summed_grad + noise
            p.grad.has_noise = True

    def scale_grad(self):
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations

    # TODO: see GradSampleModule.zero_grad()
    # TODO: actually, not calling zero_grad() after step() does break privacy accounting - add warning?
    def zero_grad(self, set_to_none: bool = False):
        for p in self.params:
            if hasattr(p, "grad_sample"):
                del p.grad_sample
            if hasattr(p, "summed_grad"):
                del p.summed_grad

        self.optimizer.zero_grad(set_to_none)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        self.accumulate()
        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        return self.optimizer.step(closure)

    # TODO: wrap the rest of optim.Optimizer interface
