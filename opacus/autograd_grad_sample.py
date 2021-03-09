#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
r"""
*Based on* https://github.com/cybertronai/autograd-hacks

This module provides functions to capture per-sample gradients by using hooks.

Notes:
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
from opacus.layers.dp_lstm import LSTMLinear

from .supported_layers_grad_samplers import _supported_layers_grad_samplers
from .utils.module_inspection import get_layer_type, requires_grad


# work-around for https://github.com/pytorch/pytorch/issues/25723
_hooks_disabled: bool = False


def add_hooks(model: nn.Module, loss_reduction: str = "mean", batch_first: bool = True):
    r"""
    Adds hooks to model to save activations and backprop values.
    The hooks will

    1. save activations into ``param.activations`` during forward pass.
    2. compute per-sample gradients and save them in ``param.grad_sample`` during backward pass.

    Args:
        model: Model to which hooks are added.
        loss_reduction: Indicates if the loss reduction (for aggregating the
            gradients) is a sum or a mean operation. Can take values ``sum`` or
            ``mean``.
        batch_first: Flag to indicate if the input tensor to the corresponding module
            has the first dimension represent the batch, for example of shape
            ``[batch_size, ..., ...]``. Set to True if batch appears in first
            dimension else set to False (``batch_first=False`` implies that the
            batch is always in the second dimension).
    """
    if hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Trying to add hooks twice to the same model")

    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if get_layer_type(layer) in _supported_layers_grad_samplers.keys():
            handles.append(layer.register_forward_hook(_capture_activations))

            handles.append(
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

    Args:
        model: Model from which hooks are to be removed.
    """
    if not hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Asked to remove hooks, but no hooks found")
    else:
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

    Args:
        layer: Layer for which we need to determine if the support for
            capturing per-sample gradients is available.

    Returns:
        Whether the ``layer`` is supported by this library.
    """
    return get_layer_type(layer) in list(_supported_layers_grad_samplers.keys()) + [
        "DPLSTM",
        "DPMultiheadAttention",
    ]


def _capture_activations(
    layer: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]
):
    r"""Forward hook handler captures and saves activations flowing into the
    ``layer`` in ``layer.activations`` during forward pass.

    Args:
        layer: Layer to capture the activations in.
        inputs: Inputs to the ``layer``.
        outputs: Outputs of the ``layer``.
    """
    layer_type = get_layer_type(layer)
    if (
        not requires_grad(layer)
        or layer_type not in _supported_layers_grad_samplers.keys()
        or not layer.training
    ):
        return

    if _hooks_disabled:
        return
    if get_layer_type(layer) not in _supported_layers_grad_samplers.keys():
        raise ValueError("Hook installed on unsupported layer")

    if not hasattr(layer, "activations"):
        layer.activations = []

    layer.activations.append(inputs[0].detach())


def _capture_backprops(
    layer: nn.Module,
    inputs: Tuple[torch.Tensor],
    outputs: Tuple[torch.Tensor],
    loss_reduction: str,
    batch_first: bool,
):
    r"""Backward hook handler captures backpropagated gradients during
    backward pass, and computes and stores per-sample gradients.

    Args:
        layer: Layer to capture gradients in.
        inputs: Gradients of the tensor on which ``.backward()`` is called with
            respect to the inputs to the ``layer``.
        outputs: Gradients of the tensor on which ``.backward()`` is called with
            respect to the outputs of the ``layer``.
        loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
            is a sum or a mean operation. Can take values ``sum`` or ``mean``.
        batch_first: Flag to indicate if the input tensor to the corresponding module
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

    Args:
        layer: Layer to capture per-sample gradients in.
        backprops: Back propagated gradients captured by the backward hook.
        loss_reduction: Indicates if the loss reduction (for aggregating the
            gradients) is a sum or a mean operation. Can take values ``sum``
            or ``mean``.
        batch_first: Flag to indicate if the input tensor to the corresponding
            module has the first dimension represent the batch, for example of
            shape ``[batch_size, ..., ...]``. Set to True if batch appears in
            first dimension else set to False (``batch_first=False`` implies
            that the batch is always in the second dimension).
    """
    layer_type = get_layer_type(layer)
    if (
        not requires_grad(layer)
        or layer_type not in _supported_layers_grad_samplers.keys()
        or not layer.training
    ):
        return

    if not hasattr(layer, "activations"):
        raise ValueError(
            f"No activations detected for {type(layer)},"
            " run forward after add_hooks(model)"
        )

    # Outside of the LSTM there is "batch_first" but not for the Linear inside the LSTM
    batch_dim = 0 if batch_first or type(layer) is LSTMLinear else 1
    if isinstance(layer.activations, list):
        A = layer.activations.pop()
    else:
        A = layer.activations

    if not hasattr(layer, "max_batch_len"):
        layer.max_batch_len = _get_batch_size(layer, A, batch_dim)

    n = layer.max_batch_len
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
        B = B.permute([batch_dim] + [x for x in range(B.dim()) if x != batch_dim])
    # compute grad sample for  individual layers
    compute_layer_grad_sample = _supported_layers_grad_samplers.get(
        get_layer_type(layer)
    )

    compute_layer_grad_sample(layer, A, B)

    if (
        not isinstance(layer.activations, list) or len(layer.activations) == 0
    ) and hasattr(layer, "max_batch_len"):
        del layer.max_batch_len


def _get_batch_size(layer: nn.Module, grad_sample: torch.Tensor, batch_dim: int) -> int:
    r"""
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over layer.activations + [grad_sample], where layer.activations is
    a list. If layer.activations is a not a list, then return grad_sample.shape[batch_dim].
    """

    max_batch_len = 0
    if isinstance(layer.activations, list):
        for out in layer.activations:
            if out.shape[batch_dim] > max_batch_len:
                max_batch_len = out.shape[batch_dim]

    max_batch_len = max(max_batch_len, grad_sample.shape[batch_dim])
    return max_batch_len
