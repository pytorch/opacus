import torch
import torch.nn as nn
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from torch.nn.utils._per_sample_grad import call_for_per_sample_grads

class GradSampleModuleExpandedWeights(AbstractGradSampleModule):

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
    ):
        if not batch_first:
            raise NotImplementedError

        if loss_reduction != "mean":
            raise NotImplementedError

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return call_for_per_sample_grads(self._module, x.shape[0], x, *args, **kwargs)
