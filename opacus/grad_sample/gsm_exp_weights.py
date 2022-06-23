import torch
import torch.nn as nn
from opacus.grad_sample.gsm_base import AbstractGradSampleModule


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

        try:
            from torch.nn.utils._per_sample_grad import call_for_per_sample_grads
            self.call_for_per_sample_grads = call_for_per_sample_grads
        except ImportError:
            raise ImportError(
                f"Requested grad_sample_mode=ew, "
                f"but found PyTorch version={torch.__version__}. "
                f"ExpandedWeights available for torch>=1.12. "
                f"Please install recent PyTorch or use grad_sample_mode=hooks"
            )

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.call_for_per_sample_grads(self._module, x.shape[0], x, *args, **kwargs)
