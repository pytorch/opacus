#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Sequence, Type, Union

import torch
import torch.nn as nn

from .grad_sample_module import GradSampleModule


def register_grad_sampler(
    target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]]
):
    """
    Registers the decorated function as the ``grad_sampler`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to compute a per-sample gradient
    of ``target_class_or_classes``. The signature of every grad_sampler is always the same:

    >>> @register_grad_sampler(nn.MyCustomClass)
    >>> def compute_grad_sample(module, activations, backprops):
    >>>    pass

    It may help you to take a look at the existing grad_samplers inside Opacus, under ``opacus.grad_sample.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            GradSampleModule.GRAD_SAMPLERS[target_class] = f
        return f

    return decorator


def create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int
) -> None:
    """
    Creates a ``grad_sample`` attribute in the given parameter, or appends to it
    if the ``grad_sample`` attribute already exists.

    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
        batch_dim: Position of the batch dimension in the shape of
            ``grad_sample``
    """

    if hasattr(param, "grad_sample"):
        param.grad_sample = torch.cat((param.grad_sample, grad_sample), batch_dim)
    else:
        param.grad_sample = grad_sample


def create_or_accumulate_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, layer: nn.Module
) -> None:
    """
    Creates a ``grad_sample`` attribute in the given parameter, or adds to it
    if the ``grad_sample`` attribute already exists.

    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
    """

    if hasattr(param, "grad_sample"):
        param.grad_sample[: grad_sample.shape[0]] += grad_sample
    else:
        max_batch_len = layer.max_batch_len
        param.grad_sample = torch.zeros(
            torch.Size([max_batch_len]) + grad_sample.shape[1:],
            device=grad_sample.device,
            dtype=grad_sample.dtype,
        )
        param.grad_sample[: grad_sample.shape[0]] = grad_sample
