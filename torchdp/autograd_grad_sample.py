#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Taken from https://github.com/cybertronai/autograd-hacks

Original license is Unlicense. We put it here for user's convenience, with
the author's permission.
"""
from functools import partial
from typing import List

import torch
import torch.nn as nn

from .supported_layers_grad_samplers import _supported_layers_grad_samplers
from .utils.module_inspection import get_layer_type, requires_grad


# work-around for https://github.com/pytorch/pytorch/issues/25723
_hooks_disabled: bool = False

# global switch to catch double backprop errors on Hessian computation
_enforce_fresh_backprop: bool = False


def add_hooks(
    model: nn.Module, loss_reduction: str = "mean", batch_first: bool = True
) -> None:
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


def remove_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks(model)
    """
    if not hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_grad_sample_hooks:
            handle.remove()
        del model.autograd_grad_sample_hooks


def disable_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """
    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks() -> None:
    """the opposite of disable_hooks()"""
    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""
    return get_layer_type(layer) in _supported_layers_grad_samplers.keys()


def _capture_activations(
    layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor
):
    """Save activations into layer.activations in forward pass"""
    if _hooks_disabled:
        return
    if get_layer_type(layer) not in _supported_layers_grad_samplers.keys():
        raise ValueError("Hook installed on unsupported layer")

    layer.activations = input[0].detach()


def _capture_backprops(
    layer: nn.Module,
    _input: torch.Tensor,
    output: torch.Tensor,
    loss_reduction: str,
    batch_first: bool,
):
    """Capture backprops in backward pass and store per-sample gradients."""

    if _hooks_disabled:
        return

    backprops = output[0].detach()
    _compute_grad_sample(layer, backprops, loss_reduction, batch_first)


def _compute_grad_sample(
    layer: nn.Module, backprops: torch.Tensor, loss_reduction: str, batch_first: bool
) -> None:
    """
    Compute per-example gradients and save them under 'param.grad_sample'.
    Must be called after loss.backprop()
    Args:
        layer: the layer for which per-sample gradients are computed
        backprops: the captured backprops
        loss_type: either "mean" or "sum" depending on whether backpropped
        loss was averaged or summed over batch
        batch_first: True is batch dimension is first
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
        B = B.permute([batch_dim] + [x for x in range(B.dim()) if x != batch_dim])
    # compute grad sample for  individual layers
    compute_layer_grad_sample = _supported_layers_grad_samplers.get(
        get_layer_type(layer)
    )
    compute_layer_grad_sample(layer, A, B)
