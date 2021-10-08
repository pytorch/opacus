import abc
from abc import ABC
from typing import Iterable, List

import torch
import torch.nn as nn


class IGradientClipper(ABC):
    @abc.abstractmethod
    def clip(self, params: List[nn.Parameter]):
        pass


class FlatGradientClipper(IGradientClipper):
    def __init__(self, max_grad_norm: float, retain_grad_sample: bool = False):
        self.max_grad_norm = max_grad_norm
        self.retain_grad_sample = retain_grad_sample

    def clip(self, params: Iterable[nn.Parameter]):
        per_param_norms = [
            p.grad_sample.view(len(p.grad_sample), -1).norm(2, dim=-1)
            for p in params
            if p.requires_grad
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
            max=1.0
        )

        for p in params:
            if not p.requires_grad:
                continue

            if not hasattr(p, "clipped_grads"):
                p.clipped_grads = []

            p.clipped_grads.append(
                torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample)
            )

            if not self.retain_grad_sample:
                del p.grad_sample
