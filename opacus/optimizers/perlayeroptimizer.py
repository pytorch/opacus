from __future__ import annotations

from typing import List, Optional

import torch
from torch.optim import Optimizer

from .optimizer import DPOptimizer


class DPPerLayerOptimizer(DPOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norms: List[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
    ):
        assert len(max_grad_norms) == len(optimizer.params)
        self.max_grad_norms = max_grad_norms
        max_grad_norm = torch.norm(torch.Tensor(self.max_grad_norms), p=2).item()
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
        )

    def attach(self, optimizer):
        self.optimizer = optimizer

    def clip_and_accumulate(self):
        for (p, max_grad_norm) in zip(self.params, self.max_grad_norms):
            per_sample_norms = p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=1)
            per_sample_clip_factor = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )
            grad = torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample)

            if hasattr(p, "summed_grad"):
                p.summed_grad += grad
            else:
                p.summed_grad = grad