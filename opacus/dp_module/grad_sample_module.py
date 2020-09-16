#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from functools import partial
from typing import Iterable, List

import torch
import torch.nn as nn

from .grad_sample_hooks import GRAD_SAMPLERS, SUPPORTED_MODULES


# work-around for https://github.com/pytorch/pytorch/issues/25723
_hooks_disabled: bool = False


class GradSampleModule(nn.Module):
    r"""
    Extends nn.Module so that its parameter tensors have an extra field called .grad_sample.
    """

    def __init__(self, m: nn.Module):
        super().__init__()
        self._module = m
        self.add_hooks()

    def forward(self, x):
        return self._module(x)

    def zero_grad(self):
        self.del_grad_sample()
        super().zero_grad()

    def del_grad_sample(self):
        """
        Deletes .grad_sample from this module's parameters.
        Why del? Normally, `zero_grad()` would do p.grad.zero_() and keep the allocation.
        Normal grads can do this, because their shape is always the same.
        Grad samples do not behave like this, because they accumulate over the batch dim.
        If you have batch_size=32 and size (12, 16) and you backprop twice, you should
        expect to have grad_samples of size [64, 12, 16]. If you backprop once more,
        then you'll get size [96, 12, 16] and so on.
        So when you zero out, you should be left with nothing so you can start over.
        """
        for p in self.parameters():
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                if p.grad_sample.grad_fn is not None:
                    p.grad_sample.detach_()
                else:
                    p.grad_sample.requires_grad_(False)

                del p.grad_sample

    def to_standard_module(self) -> nn.Module:
        """
        Returns the standard nn.Module wrapped by this, eliminating all traces
        of grad samples and hooks

        Returns:
            The wrapped module
        """
        self.del_grad_sample()
        self.remove_hooks()
        return self._module

    def add_hooks(self, loss_reduction: str = "mean", batch_first: bool = True) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call "remove_hooks(model)" to disable this.
        Args:
            model: the model to which hooks are added
            loss_type: either "mean" or "sum" depending on whether backpropped
            loss was averaged or summed over batch (default: "mean")
            batch_dim: the batch dimension (default: 0)
        """
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []

        global _hooks_disabled
        _hooks_disabled = False

        for module in self.trainable_modules():
            if self.is_supported(module):
                self._module.autograd_grad_sample_hooks.append(
                    module.register_forward_hook(_capture_activations)  # pyre-ignore
                )

                self._module.autograd_grad_sample_hooks.append(
                    module.register_backward_hook(  # pyre-ignore
                        partial(
                            _capture_backprops,
                            loss_reduction=loss_reduction,
                            batch_first=batch_first,
                        )
                    )
                )
            else:
                raise ValueError(
                    f"The submodule {module} wants a gradient but is not currently supported. "
                    "Consider changing it, or freezing it by setting .requires_grad to False "
                    "for each of its parameters (this will block gradient propagation)."
                )

    def remove_hooks(self) -> None:
        """
        Removes hooks added by add_hooks()
        """
        if not hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Asked to remove hooks, but no hooks found")
        else:
            for handle in self._module.autograd_grad_sample_hooks:  # pyre-ignore
                handle.remove()
            del self._module.autograd_grad_sample_hooks

    def disable_hooks(self) -> None:
        r"""
        Globally disable all hooks installed by this library.

        Why is this needed? As per https://github.com/pytorch/pytorch/issues/25723, there is
        a bug in Autograd that makes removing hooks do nothing if the graph was already
        constructed. For this reason, we have this method to at least turn them off.
        """
        global _hooks_disabled
        _hooks_disabled = True

    def enable_hooks(self) -> None:
        r"""
        The opposite of disable_hooks(). Hooks are always enabled unless you explicitly
        disable them so you don't need to call this unless you want to re-enable them.
        """
        global _hooks_disabled
        _hooks_disabled = False

    def parametrized_modules(self) -> Iterable[nn.Module]:
        """
        Recursively iterates over all submodules, returning those that
        have parameters (as opposed to "wrapper modules" that just organize modules).
        """
        yield from (
            m
            for m in self._module.modules()
            if any(p is not None for p in m.parameters(recurse=False))
        )

    def trainable_modules(self) -> Iterable[nn.Module]:
        """
        Recursively iterates over all submodules, returning those that
        have parameters and are trainable (ie they want a grad).
        """
        yield from (
            m
            for m in self.parametrized_modules()
            if any(p.requires_grad for p in m.parameters())
        )

    def is_supported(self, module: nn.Module) -> bool:
        """Check if this module is supported"""
        return type(module) in SUPPORTED_MODULES


def requires_grad(module: nn.Module, recurse: bool = False):
    return any(p.requires_grad for p in module.parameters(recurse))


def _capture_activations(
    module: nn.Module, forward_input: List[torch.Tensor], _forward_output: torch.Tensor
):
    """Save activations into module.activations in forward pass"""
    if _hooks_disabled:
        return
    if type(module) not in SUPPORTED_MODULES:
        raise ValueError("Hook installed on unsupported module")

    module.activations = forward_input[0].detach()  # pyre-ignore


def _capture_backprops(
    module: nn.Module,
    _forward_input: torch.Tensor,
    forward_output: torch.Tensor,
    loss_reduction: str,
    batch_first: bool,
):
    """Capture backprops in backward pass and store per-sample gradients."""

    if _hooks_disabled:
        return

    backprops = forward_output[0].detach()
    _compute_grad_sample(module, backprops, loss_reduction, batch_first)


def _compute_grad_sample(
    module: nn.Module, backprops: torch.Tensor, loss_reduction: str, batch_first: bool
) -> None:
    """
    Compute per-example gradients and save them under 'param.grad_sample'.
    Must be called after loss.backprop()
    Args:
        module: the module for which per-sample gradients are computed
        backprops: the captured backprops
        loss_type: either "mean" or "sum" depending on whether backpropped
        loss was averaged or summed over batch
        batch_first: True is batch dimension is first
    """
    if not requires_grad(module) or type(module) not in SUPPORTED_MODULES:
        return

    if not hasattr(module, "activations"):
        raise ValueError(
            f"No activations detected for {type(module)},"
            " run forward after add_hooks(model)"
        )

    batch_dim = 0 if batch_first else 1

    # pyre-fixme[16]: `Module` has no attribute `activations`.
    A = module.activations
    n = A.shape[batch_dim]
    if loss_reduction == "mean":
        B = backprops * n
    elif loss_reduction == "sum":
        B = backprops
    else:
        raise ValueError(
            f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported"
        )

    # rearrange the blob dimensions
    if batch_dim != 0:
        A = A.permute([batch_dim] + [x for x in range(A.dim()) if x != batch_dim])
        # pyre-fixme[6]: Expected `int` for 1st param but got `List[int]`.
        B = B.permute([batch_dim] + [x for x in range(B.dim()) if x != batch_dim])
    grad_sample_fn = GRAD_SAMPLERS[type(module)]
    grad_sample_fn(module, A, B)
