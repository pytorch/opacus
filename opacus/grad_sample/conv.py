#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.tensor_utils import unfold2d, unfold3d

from .utils import register_grad_sampler


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for convolutional layers

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    n = activations.shape[0]
    # get activations and backprops in shape depending on the Conv layer
    if type(layer) == nn.Conv2d:
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    elif type(layer) == nn.Conv1d:
        activations = activations.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        activations = torch.nn.functional.unfold(
            activations,
            kernel_size=(1, layer.kernel_size[0]),
            padding=(0, layer.padding[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
    elif type(layer) == nn.Conv3d:
        activations = unfold3d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    backprops = backprops.reshape(n, -1, activations.shape[-1])
    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    grad_sample = torch.einsum("noq,npq->nop", backprops, activations)
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

    ret = {layer.weight: grad_sample.view(shape)}
    if layer.bias is not None:
        ret[layer.bias] = torch.sum(backprops, dim=2)

    return ret
