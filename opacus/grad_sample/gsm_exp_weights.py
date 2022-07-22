import torch
import torch.nn as nn
from opacus.grad_sample.gsm_base import AbstractGradSampleModule


COMPATIBILITY_API_CUTOFF_VERSION = "1.13.0.dev"


class GradSampleModuleExpandedWeights(AbstractGradSampleModule):
    """
    ExpandedWeights-based implementation of AbstractGradSampleModule

    Computes per-sample gradients using PyTorch built-in mechanism of ExpandedWeights.
    See README.md for more details
    """

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
        if torch.__version__ >= COMPATIBILITY_API_CUTOFF_VERSION:
            return self.call_for_per_sample_grads(
                module=self._module,
                batch_size=x.shape[0],
                loss_reduction=self.loss_reduction,
            )(x, *args, **kwargs)
        else:
            return self.call_for_per_sample_grads(
                module=self._module, batch_size=x.shape[0], args=(x, *args), **kwargs
            )
