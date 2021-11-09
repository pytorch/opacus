#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Union, Tuple

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import unfold3d

from .utils import register_grad_sampler

# def unfoldnd(input, kernel_size: tuple, stride: tuple):
#     batch, channels, *dims = input.shape
#     n_frames = [
#         (max(le, k) - k) // s + 1 for le, k, s in zip(dims, kernel_size, stride)]
#     tgt_lengths = [
#         (f - 1) * s + k
#         for f, k, s in zip(n_frames, kernel_size, stride)]
#     indexing = tuple([slice(None), slice(None)] + [slice(le) for le in tgt_lengths])
#     input = input[indexing].contiguous()
#     ostrides = list(input.stride())
#     strides = []
#     current_stride = 1
#     for s in stride[::-1]:
#         strides += [current_stride, s * current_stride]
#         current_stride *= s
#     strides = strides[::-1]
#     strides = ostrides[:2] + strides
#     out_shapes = [batch, channels]
#     for f, k in zip(n_frames, kernel_size):
#         out_shapes += [f, k]
#     print(strides, out_shapes)
#     out = input.as_strided(out_shapes, strides)
    
#     return out

def unfold2d(input, kernel_size: Tuple[int, int], padding: Tuple[int, int], stride: Tuple[int, int], dilation: Tuple[int, int]):
    # print("input shape", input.shape)
    *shape, H, W = input.shape
    H_effective = (H + 2 * padding[0] - (kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1))) // stride[0] + 1
    W_effective = (W + 2 * padding[1] - (kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1))) // stride[1] + 1
    input = F.pad(input, (padding[0], padding[0], padding[1], padding[1]))
    *shape_pad, H_pad, W_pad = input.shape
    # print(input.shape, input.stride())
    strides = list(input.stride())
    strides = strides[:-2] + [W_pad * dilation[0], dilation[1], W_pad*stride[0], stride[1]]
    # print("STRIDES", strides)
    # print("SHAPE", shape + [kernel_size[0], kernel_size[1], H_effective, W_effective])
    out = input.as_strided(shape + [kernel_size[0], kernel_size[1], H_effective, W_effective], strides)

    return out.reshape(input.size(0), -1, H_effective * W_effective)

def unfold1d(input, kernel_size: int, stride: int, padding):
    *shape, length = input.shape
    n_frames = (max(length, kernel_size) - kernel_size) // stride + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    input = F.pad(input[..., :tgt_length], padding).contiguous()
    strides = list(input.stride())
    strides = strides[:-1] + [stride, 1]
    print("STRIDES", strides)
    print("SHAPE", shape + [n_frames, kernel_size])
    out = input.as_strided(shape + [n_frames, kernel_size], strides)
    return out.transpose(-1, -2)


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv2d, nn.Conv1d],
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
    # start = time.time()
    n = activations.shape[0]
    # get A and B in shape depending on the Conv layer
    if type(layer) == nn.Conv2d:
        # activations3 = torch.nn.functional.unfold(
        #     activations,
        #     layer.kernel_size,
        #     padding=layer.padding,
        #     stride=layer.stride,
        #     dilation=layer.dilation,
        # )
        activations3 = unfold2d(activations, layer.kernel_size, layer.padding, layer.stride, layer.dilation)
        backprops = backprops.view(n, -1, activations3.shape[-1])
        # try:
        #     assert torch.norm(activations3 - activations2) < 1e-4
        # except:
        #     import ipdb;ipdb.set_trace()
        activations = activations3
    elif type(layer) == nn.Conv1d:

        activations = unfold1d(
            activations,
            kernel_size=layer.kernel_size[0],
            stride=layer.stride[0]
        )
        activations = activations.reshape(n, -1, activations.shape[-1])

        # unfold doesn't work for 3D tensors; so force it to be 4D
        # set arguments to tuples with appropriate second element
        # activations = activations.unsqueeze(-2)  # add the H dimension
        # activations = torch.nn.functional.unfold(
        #     activations,
        #     (1, layer.kernel_size[0]),
        #     padding=(0, layer.padding[0]),
        #     stride=(1, layer.stride[0]),
        #     dilation=(1, layer.dilation[0]),
        # )
        # print(activations.shape)
        backprops = backprops.view(n, -1, activations.shape[-1])
    elif type(layer) == nn.Conv3d:
        activations = unfold3d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
        backprops = backprops.reshape(n, -1, activations.shape[-1])

    # end_unfold = time.time()
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

    # end = time.time()
    # print(f"Timings:\tunfold:{end_unfold - start:.2e}\trest:{end - end_unfold:.2e}")

    return ret
