#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.functional import F

from .utils.module_inspection import get_layer_type
from .utils.tensor_utils import sum_over_all_but_batch_and_last_n


def _create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int
) -> None:
    """
    Create a 'grad_sample' attribute in the given parameter, or append to it
    if the 'grad_sample' attribute already exists.
    """

    if hasattr(param, "grad_sample"):
        # pyre-fixme[16]: `Tensor` has no attribute `grad_sample`.
        param.grad_sample = torch.cat((param.grad_sample, grad_sample), batch_dim)
    else:
        param.grad_sample = grad_sample


def _compute_linear_grad_sample(layer, A, B, batch_dim=0):
    gs = torch.einsum("n...i,n...j->n...ij", B, A)
    _create_or_extend_grad_sample(
        layer.weight, torch.einsum("n...ij->nij", gs), batch_dim
    )
    if layer.bias is not None:
        _create_or_extend_grad_sample(
            layer.bias, torch.einsum("n...k->nk", B), batch_dim
        )


def _compute_sequence_bias_grad_sample(layer, A, B, batch_dim=0):
    _create_or_extend_grad_sample(layer.bias, B[:, -1], batch_dim)


def _compute_norm_grad_sample(layer, A, B, batch_dim=0):
    layer_type = get_layer_type(layer)
    if layer_type == "LayerNorm":
        _create_or_extend_grad_sample(
            layer.weight,
            sum_over_all_but_batch_and_last_n(
                F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
                layer.weight.dim(),
            ),
            batch_dim,
        )
        _create_or_extend_grad_sample(
            layer.bias,
            sum_over_all_but_batch_and_last_n(B, layer.bias.dim()),
            batch_dim,
        )
    elif layer_type == "GroupNorm":
        gs = F.group_norm(A, layer.num_groups, eps=layer.eps) * B
        _create_or_extend_grad_sample(
            layer.weight, torch.einsum("ni...->ni", gs), batch_dim
        )
        if layer.bias is not None:
            _create_or_extend_grad_sample(
                layer.bias, torch.einsum("ni...->ni", B), batch_dim
            )
    elif layer_type in {"InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"}:
        gs = F.instance_norm(A, eps=layer.eps) * B
        _create_or_extend_grad_sample(
            layer.weight, torch.einsum("ni...->ni", gs), batch_dim
        )
        if layer.bias is not None:
            _create_or_extend_grad_sample(
                layer.bias, torch.einsum("ni...->ni", B), batch_dim
            )


def _compute_dplstm_grad_sample(layer, A, B, batch_dim=0):
    lstm_params = [
        layer.weight_ih_l0,
        layer.weight_hh_l0,
        layer.bias_ih_l0,
        layer.bias_hh_l0,
    ]
    lstm_out_dim = layer.hidden_size

    x = torch.unbind(A, dim=1)
    hooks_delta = torch.unbind(B, dim=1)

    SEQ_LENGTH = len(x)
    BATCH_SIZE = B.shape[0]

    h_init = torch.zeros(1, BATCH_SIZE, lstm_out_dim, device=A.device)
    c_init = torch.zeros(1, BATCH_SIZE, lstm_out_dim, device=A.device)

    delta_h = {}
    delta_h[SEQ_LENGTH - 1] = 0
    f_last = 0
    dc_last = 0

    for t in range(SEQ_LENGTH - 1, -1, -1):
        f_next = f_last if t == SEQ_LENGTH - 1 else layer.cells[t + 1].f_t
        dc_next = dc_last if t == SEQ_LENGTH - 1 else layer.cells[t + 1].dc_t
        c_prev = c_init if t == 0 else layer.cells[t - 1].c_t
        delta_h[t - 1] = layer.cells[t].backward(
            x[t], delta_h[t], hooks_delta[t], f_next, dc_next, c_prev
        )

    grad_sample = {param: 0 for param in lstm_params}

    for t in range(0, SEQ_LENGTH):
        h_prev = h_init[0, :] if t == 0 else layer.cells[t - 1].h_t[0, :]
        grad_sample[layer.weight_ih_l0] += torch.einsum(
            "ij,ik->ijk", layer.cells[t].dgates_t, x[t]
        )
        grad_sample[layer.weight_hh_l0] += torch.einsum(
            "ij,ik->ijk", layer.cells[t].dgates_t, h_prev
        )
        grad_sample[layer.bias_ih_l0] += layer.cells[t].dgates_t
        grad_sample[layer.bias_hh_l0] += layer.cells[t].dgates_t

    for param, grad_value in grad_sample.items():
        _create_or_extend_grad_sample(param, grad_value, batch_dim)


def _compute_conv_grad_sample(layer, A, B, batch_dim=0):
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
        _create_or_extend_grad_sample(
            layer.weight, grad_sample.reshape(shape), batch_dim
        )
    except Exception as e:
        raise type(e)(
            f"{e} There is probably a problem with {layer_type}.groups"
            + "It should be either 1 or in_channel"
        )
    if layer.bias is not None:
        _create_or_extend_grad_sample(layer.bias, torch.sum(B, dim=2), batch_dim)


def _compute_embedding_grad_sample(layer, A, B, batch_dim=0):
    one_hot = F.one_hot(A, num_classes=layer.weight.shape[0])
    gs = torch.einsum("n...i,n...j->n...ij", one_hot, B)

    _create_or_extend_grad_sample(
        layer.weight, torch.einsum("n...ij->nij", gs), batch_dim
    )


_supported_layers_grad_samplers = {
    "Embedding": _compute_embedding_grad_sample,
    "Linear": _compute_linear_grad_sample,
    "Conv2d": _compute_conv_grad_sample,
    "Conv1d": _compute_conv_grad_sample,
    "LayerNorm": _compute_norm_grad_sample,
    "GroupNorm": _compute_norm_grad_sample,
    "InstanceNorm1d": _compute_norm_grad_sample,
    "InstanceNorm2d": _compute_norm_grad_sample,
    "InstanceNorm3d": _compute_norm_grad_sample,
    "SequenceBias": _compute_sequence_bias_grad_sample,
    "DPLSTM": _compute_dplstm_grad_sample,
}  # Supported layer class types
