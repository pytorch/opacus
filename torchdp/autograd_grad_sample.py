#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Taken from https://github.com/cybertronai/autograd-hacks

Original license is Unlicense. We put it here for user's convenience, with
the author's permission.
"""

from typing import List

import torch
import torch.nn as nn

from .supported_layers_grad_samplers import _supported_layers_grad_samplers
from .utils import get_layer_type, requires_grad


# work-around for https://github.com/pytorch/pytorch/issues/25723
_hooks_disabled: bool = False

# global switch to catch double backprop errors on Hessian computation
_enforce_fresh_backprop: bool = False


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.
    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.
    Call "remove_hooks(model)" to disable this.
    Args:
        model:
    """
    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if get_layer_type(layer) in _supported_layers_grad_samplers.keys():
            handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_backward_hook(_capture_backprops))

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


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _enforce_fresh_backprop

    if _hooks_disabled:
        return

    if _enforce_fresh_backprop:
        if hasattr(layer, "backprops_list"):
            raise ValueError(
                f"Seeing result of previous backprop, "
                f"use clear_backprops(model) to clear"
            )
        _enforce_fresh_backprop = False

    if not hasattr(layer, "backprops_list"):
        layer.backprops_list = []
    layer.backprops_list.append(output[0].detach())


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, "backprops_list"):
            del layer.backprops_list


def _check_layer_sanity(layer):
    if not hasattr(layer, "activations"):
        raise ValueError(
            f"No activations detected for {type(layer)},"
            " run forward after add_hooks(model)"
        )
    if not hasattr(layer, "backprops_list"):
        raise ValueError("No backprops detected, run backward after add_hooks(model)")
    if len(layer.backprops_list) != 1:
        raise ValueError(
            "Multiple backprops detected, make sure to call clear_backprops(model)"
        )


def compute_grad_sample(
    model: nn.Module, loss_type: str = "mean", batch_dim: int = 0
) -> None:
    """
    Compute per-example gradients and save them under 'param.grad_sample'.
    Must be called after loss.backprop()
    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped
        loss was averaged or summed over batch
    """
    if loss_type not in ("sum", "mean"):
        raise ValueError(f"loss_type = {loss_type}. Only 'sum' and 'mean' supported")
    for layer in model.modules():
        layer_type = get_layer_type(layer)
        if (
            not requires_grad(layer)
            or layer_type not in _supported_layers_grad_samplers.keys()
        ):
            continue

        _check_layer_sanity(layer)

        A = layer.activations
        n = A.shape[batch_dim]
        if loss_type == "mean":
            B = layer.backprops_list[0] * n
        else:  # loss_type == 'sum':
            B = layer.backprops_list[0]
        # rearrange the blob dimensions
        if batch_dim != 0:
            A = A.permute([batch_dim] + [x for x in range(A.dim()) if x != batch_dim])
            B = B.permute([batch_dim] + [x for x in range(B.dim()) if x != batch_dim])
        # compute grad sample for  individual layers
        compute_layer_grad_sample = _supported_layers_grad_samplers.get(
            get_layer_type(layer)
        )
        compute_layer_grad_sample(layer, A, B)
