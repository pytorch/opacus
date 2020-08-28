#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
r"""
*Based on* https://github.com/cybertronai/autograd-hacks

This module provides functions to capture per-sample gradients by using hooks.

Notes
-----
The ``register_backward_hook()`` function has a known issue being tracked at
https://github.com/pytorch/pytorch/issues/598. However, it is the only known
way of implementing this as of now (your suggestions and contributions are
very welcome). The behaviour has been verified to be correct for the layers
currently supported by opacus.
"""

from functools import partial
from typing import Tuple

import torch
import torch.nn as nn

from .supported_layers_grad_samplers import _supported_layers_grad_samplers
from .utils.module_inspection import get_layer_type, requires_grad


# work-around for https://github.com/pytorch/pytorch/issues/25723
_hooks_disabled: bool = False


def add_hooks(model: nn.Module, loss_reduction: str = "mean", batch_first: bool = True):
    r"""
    Adds hooks to model to save activations and backprop values.
    The hooks will

    1. save activations into ``param.activations`` during forward pass.
    2. compute per-sample gradients and save them in ``param.grad_sample``
    during backward pass.

    Parameters
    ----------
    model: nn.Module
        Model to which hooks are added.
    loss_reduction: str
        Indicates if the loss reduction (for aggregating the gradients)
        is a sum or a mean operation. Can take values ``sum`` or ``mean``.
        Default value is ``mean``.
    batch_first: bool
        Flag to indicate if the input tensor to the corresponding module
        has the first dimension represent the batch, for example of shape
        ``[batch_size, ..., ...]``. Set to True if batch appears in first
        dimension else set to False (``batch_first=False`` implies that the
        batch is always in the second dimension).
        Default value is ``True``.
    """
    if hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Trying to add hooks twice to the same model")

    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if get_layer_type(layer) in _supported_layers_grad_samplers.keys():
            # pyre-fixme[16]: `Module` has no attribute `register_forward_hook`.
            handles.append(layer.register_forward_hook(_capture_activations))

            handles.append(
                # pyre-fixme[16]: `Module` has no attribute `register_backward_hook`.
                layer.register_backward_hook(
                    partial(
                        _capture_backprops,
                        loss_reduction=loss_reduction,
                        batch_first=batch_first,
                    )
                )
            )

    model.__dict__.setdefault("autograd_grad_sample_hooks", []).extend(handles)


def remove_hooks(model: nn.Module):
    r"""
    Removes hooks added by ``add_hooks()``.

    Parameters
    ----------
    model: nn.Module
        Model from which hooks are to be removed.
    """
    if not hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Asked to remove hooks, but no hooks found")
    else:
        # pyre-fixme[16]: `Module` has no attribute `autograd_grad_sample_hooks`.
        for handle in model.autograd_grad_sample_hooks:
            handle.remove()
        del model.autograd_grad_sample_hooks


def disable_hooks():
    r"""
    Globally disables all hooks installed by this library.
    """
    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks():
    r"""
    Globally enables all hooks installed by this library.
    """
    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    r"""Checks if the ``layer`` is supported by this library.

    Parameters
    ----------
    layer: nn.Module
        Layer for which we need to determine if the support for capturing
        per-sample gradients is available.

    Returns
    -------
    bool
        Whether the ``layer`` is supported by this library.
    """
    return get_layer_type(layer) in _supported_layers_grad_samplers.keys()


def _capture_activations(
    layer: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]
):
    r"""Forward hook handler captures and saves activations flowing into the
    ``layer`` in ``layer.activations`` during forward pass.

    Parameters
    ----------
    layer: nn.Module
        Layer to capture the activations in.
    inputs: List[torch.Tensor]
        Inputs to the ``layer``.
    outputs: List[torch.Tensor]
        Outputs of the ``layer``.
    """
    if _hooks_disabled:
        return
    if get_layer_type(layer) not in _supported_layers_grad_samplers.keys():
        raise ValueError("Hook installed on unsupported layer")

    # pyre-fixme[16]: `Module` has no attribute `activations`.
    layer.activations = inputs[0].detach()


def _capture_backprops(
    layer: nn.Module,
    inputs: Tuple[torch.Tensor],
    outputs: Tuple[torch.Tensor],
    loss_reduction: str,
    batch_first: bool,
):
    r"""Backward hook handler captures backpropagated gradients during
    backward pass, and computes and stores per-sample gradients.

    Parameters
    ----------
    layer: nn.Module
        Layer to capture gradients in.
    inputs: List[torch.Tensor]
        Gradients of the tensor on which ``.backward()`` is called with respect
        to the inputs to the ``layer``.
    outputs: List[torch.Tensor]
        Gradients of the tensor on which ``.backward()`` is called with respect
        to the outputs of the ``layer``.
    loss_reduction: str
        Indicates if the loss reduction (for aggregating the gradients)
        is a sum or a mean operation. Can take values ``sum`` or ``mean``.
    batch_first: bool
        Flag to indicate if the input tensor to the corresponding module
        has the first dimension represent the batch, for example of shape
        ``[batch_size, ..., ...]``. Set to True if batch appears in first
        dimension else set to False (``batch_first=False`` implies that the
        batch is always in the second dimension).
    """

    if _hooks_disabled:
        return

    backprops = outputs[0].detach()
    _compute_grad_sample(layer, backprops, loss_reduction, batch_first)


def _compute_grad_sample(
    layer: nn.Module, backprops: torch.Tensor, loss_reduction: str, batch_first: bool
):
    r"""Computes per-sample gradients with respect to the parameters of the
    ``layer`` (if supported), and saves them in ``param.grad_sample``.

    Parameters
    ----------
    layer: nn.Module
        Layer to capture per-sample gradients in.
    backprops: torch.Tensor
        Back propagated gradients captured by the backward hook.
    loss_reduction: str
        Indicates if the loss reduction (for aggregating the gradients)
        is a sum or a mean operation. Can take values ``sum`` or ``mean``.
    batch_first: bool
        Flag to indicate if the input tensor to the corresponding module
        has the first dimension represent the batch, for example of shape
        ``[batch_size, ..., ...]``. Set to True if batch appears in first
        dimension else set to False (``batch_first=False`` implies that the
        batch is always in the second dimension).
    """
    layer_type = get_layer_type(layer)
    if (
        not requires_grad(layer)
        or layer_type not in _supported_layers_grad_samplers.keys()
    ):
        return

    if not hasattr(layer, "activations"):
        raise ValueError(
            f"No activations detected for {type(layer)},"
            " run forward after add_hooks(model)"
        )

    batch_dim = 0 if batch_first else 1

    # pyre-fixme[16]: `Module` has no attribute `activations`.
    A = layer.activations
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
    # compute grad sample for  individual layers
    compute_layer_grad_sample = _supported_layers_grad_samplers.get(
        get_layer_type(layer)
    )
    compute_layer_grad_sample(layer, A, B)
