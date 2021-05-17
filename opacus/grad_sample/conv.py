#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.tensor_utils import unfold3d

from .utils import create_or_extend_grad_sample, register_grad_sampler


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv2d, nn.Conv1d],
    A: torch.Tensor,
    B: torch.Tensor,
    batch_dim: int = 0,
) -> None:
    """
    Computes per sample gradients for convolutional layers

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    n = A.shape[0]
    # get A and B in shape depending on the Conv layer
    if type(layer) == nn.Conv2d:
        A = torch.nn.functional.unfold(
            A,
            layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
        B = B.reshape(n, -1, A.shape[-1])
    elif type(layer) == nn.Conv1d:
        # unfold doesn't work for 3D tensors; so force it to be 4D
        A = A.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        A = torch.nn.functional.unfold(
            A,
            (1, layer.kernel_size[0]),
            padding=(0, layer.padding[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
        B = B.reshape(n, -1, A.shape[-1])
    elif type(layer) == nn.Conv3d:
        A = unfold3d(
            A,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
        B = B.reshape(n, -1, A.shape[-1])

    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    grad_sample = torch.einsum("noq,npq->nop", B, A)
    # rearrange the above tensor and extract diagonals.
    grad_sample = grad_sample.view(
        n,
        layer.groups,
        -1,
        layer.groups,
        int(layer.in_channels / layer.groups),
        np.prod(layer.kernel_size),
    )
    grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
    shape = [n] + list(layer.weight.shape)

    create_or_extend_grad_sample(layer.weight, grad_sample.view(shape), batch_dim)

    if layer.bias is not None:
        create_or_extend_grad_sample(layer.bias, torch.sum(B, dim=2), batch_dim)
