from __future__ import annotations

from typing import List, Optional

import torch
from opacus.optimizers.utils import params
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
        for (p, max_grad_norm) in zip(self.params, self.max_grad_norms):
            _check_processed_flag(p.grad_sample)

            per_sample_norms = p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=1)
            per_sample_clip_factor = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )
            grad = torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)
