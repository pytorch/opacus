from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers.utils import params
from torch import nn
from torch.optim import Optimizer


def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True


def _check_processed_flag_tensor(x: torch.Tensor):
    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients hasn't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )


def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)


def _generate_noise(
    std: float, reference: torch.Tensor, generator=None
) -> torch.Tensor:
    if std > 0:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
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
        generator=None,
    ):
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(f"Unexpected value for loss_reduction: {loss_reduction}")

        if loss_reduction == "mean" and expected_batch_size is None:
            raise ValueError(
                "You must provide expected batch size of the loss reduction is mean"
            )

        self.original_optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction
        self.expected_batch_size = expected_batch_size
        self.step_hook = None
        self.generator = generator

        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self._step_skip_queue = []
        self._is_last_step_skipped = False

        for p in self.params:
            p.summed_grad = None

    def signal_skip_step(self, do_skip=True):
        self._step_skip_queue.append(do_skip)

    def _check_skip_next_step(self):
        if self._step_skip_queue:
            return self._step_skip_queue.pop(0)
        else:
            return False

    @property
    def params(self) -> List[nn.Parameter]:
        return params(self)

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
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
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
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
            )
            p.grad = p.summed_grad + noise

            _mark_as_processed(p.summed_grad)

    def scale_grad(self):
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations

    def zero_grad(self, set_to_none: bool = False):
        for p in self.params:
            p.grad_sample = None

            if not self._is_last_step_skipped:
                p.summed_grad = None

        self.original_optimizer.zero_grad(set_to_none)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        # TODO: handle closure call - we should do it before pre_step()
        if self.pre_step():
            return self.original_optimizer.step(closure)
        else:
            return None

    def __repr__(self):
        return self.original_optimizer.__repr__()

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.original_optimizer.load_state_dict(state_dict)
