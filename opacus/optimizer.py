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


def _get_flat_grad_sample(p: torch.Tensor):
    if not hasattr(p, "grad_sample"):
        raise ValueError(
            "Per sample gradient not found. Are you using GradSampleModule?"
        )
    if isinstance(p.grad_sample, torch.Tensor):
        return p.grad_sample
    elif isinstance(p.grad_sample, list):
        return torch.cat(p.grad_sample, dim=0)
    else:
        raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")


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

        self._is_skip_next_step = False
        self._is_last_step_skipped = False

    def skip_next_step(self):
        self._is_skip_next_step = True

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
            ret.append(_get_flat_grad_sample(p))
        return ret

    @property
    def accumulated_iterations(self) -> int:
        vals = []
        for p in self.params:
            if not hasattr(p, "grad_sample"):
                raise ValueError(
                    "Per sample gradient not found. Are you using GradSampleModule?"
                )
            if isinstance(p.grad_sample, torch.Tensor):
                vals.append(1)
            elif isinstance(p.grad_sample, list):
                vals.append(len(p.grad_sample))
            else:
                raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        if len(set(vals)) > 1:
            raise ValueError(
                "Number of accumulated steps is inconsistent across parameters"
            )
        return vals[0]

    def attach_step_hook(self, fn: Callable[[DPOptimizer], None]):
        self.step_hook = fn

    def clip_and_accumulate(self):
        per_param_norms = [
            x.view(len(x), -1).norm(2, dim=-1) for x in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        for p in self.params:
            grad_sample = _get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if hasattr(p, "summed_grad"):
                p.summed_grad += grad
            else:
                p.summed_grad = grad

    def add_noise(self):
        for p in self.params:
            noise = _generate_noise(
                self.noise_multiplier * self.max_grad_norm, p.summed_grad
            )
            p.grad = p.summed_grad + noise

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

            if hasattr(p, "summed_grad") and not self._is_last_step_skipped:
                del p.summed_grad

        self.optimizer.zero_grad(set_to_none)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        self.pre_step()

        if self._is_skip_next_step:
            self._is_skip_next_step = False
            self._is_last_step_skipped = True
            return None

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return self.optimizer.step(closure)

    # TODO: potentially refactor to decouple memory wins from accounting/averaging
    # TODO: We can potentially track virtual steps automatically (through GSM.forward() or empty activatons lists)
    def pre_step(self):
        self.clip_and_accumulate()

    # TODO: wrap the rest of optim.Optimizer interface
