#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.functional import F

from .utils import get_layer_type, sum_over_all_but_batch_and_last_n


def _compute_linear_grad_sample(layer, A, B):
    gs = torch.einsum("n...i,n...j->n...ij", B, A)
    layer.weight.grad_sample = torch.einsum("n...ij->nij", gs)
    if layer.bias is not None:
        layer.bias.grad_sample = torch.einsum("n...k->nk", B)


def _compute_sequence_bias_grad_sample(layer, A, B):
    layer.bias.grad_sample = B[:, -1]


def _compute_norm_grad_sample(layer, A, B):
    layer_type = get_layer_type(layer)
    if layer_type == "LayerNorm":
        layer.weight.grad_sample = sum_over_all_but_batch_and_last_n(
            F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
            layer.weight.dim(),
        )
        layer.bias.grad_sample = sum_over_all_but_batch_and_last_n(B, layer.bias.dim())
    elif layer_type == "GroupNorm":
        gs = F.group_norm(A, layer.num_groups, eps=layer.eps) * B
        layer.weight.grad_sample = torch.einsum("ni...->ni", gs)
        if layer.bias is not None:
            layer.bias.grad_sample = torch.einsum("ni...->ni", B)
    elif layer_type in {"InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"}:
        gs = F.instance_norm(A, eps=layer.eps) * B
        layer.weight.grad_sample = torch.einsum("ni...->ni", gs)
        if layer.bias is not None:
            layer.bias.grad_sample = torch.einsum("ni...->ni", B)


def _compute_conv_grad_sample(layer, A, B):
    n = A.shape[0]
    layer_type = get_layer_type(layer)
    # get A and B in shape depending on the Conv layer
    if layer_type == "Conv2d":
        A = torch.nn.functional.unfold(
            A, layer.kernel_size, padding=layer.padding, stride=layer.stride
        )
        B = B.reshape(n, -1, A.shape[-1])
    elif layer_type == "Conv1d":
        # unfold doesn't work for 3D tensors; so force it to be 4D
        A = A.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        A = torch.nn.functional.unfold(
            A,
            (1, layer.kernel_size[0]),
            padding=(0, layer.padding[0]),
            stride=(1, layer.stride[0]),
        )
        B = B.reshape(n, -1, A.shape[-1])
    try:
        # n=batch_sz; o=num_out_channels; p=num_in_channels*kernel_sz
        grad_sample = (
            torch.einsum("noq,npq->nop", B, A)
            if layer.groups == 1
            else torch.einsum("njk,njk->nj", B, A)
        )
        shape = [n] + list(layer.weight.shape)
        layer.weight.grad_sample = grad_sample.reshape(shape)
    except Exception as e:
        raise type(e)(
            f"{e} There is probably a problem with {layer_type}.groups"
            + "It should be either 1 or in_channel"
        )
    if layer.bias is not None:
        layer.bias.grad_sample = torch.sum(B, dim=2)


_supported_layers_grad_samplers = {
    "Linear": _compute_linear_grad_sample,
    "Conv2d": _compute_conv_grad_sample,
    "Conv1d": _compute_conv_grad_sample,
    "LayerNorm": _compute_norm_grad_sample,
    "GroupNorm": _compute_norm_grad_sample,
    "InstanceNorm1d": _compute_norm_grad_sample,
    "InstanceNorm2d": _compute_norm_grad_sample,
    "InstanceNorm3d": _compute_norm_grad_sample,
    "SequenceBias": _compute_sequence_bias_grad_sample,
}  # Supported layer class types
