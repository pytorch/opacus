from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .optimizer import DPOptimizer


class DistributedDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` compatible with
    distributed data processing
    """

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
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        # Noise only gets added to the first worker
        if self.rank == 0:
            super().add_noise()
        else:
            for p in self.params:
                p.grad = p.summed_grad

    def reduce_gradients(self):
        for p in self.params:
            if not p.requires_grad:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            if self.loss_reduction == "mean":
                p.grad /= self.world_size

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[torch.Tensor]:
        if self.pre_step():
            self.reduce_gradients()
            return self.original_optimizer.step(closure)
        else:
            return None
