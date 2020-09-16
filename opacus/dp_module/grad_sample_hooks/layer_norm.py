#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.functional import F

from .utils import create_or_extend_grad_sample


def compute_grad_sample(module, A, B, batch_dim=0):
    weight_gs = F.layer_norm(A, module.normalized_shape, eps=module.eps) * B
    weight_gs = sum_over_all_but_batch_and_last_n(weight_gs, module.weight.dim())
    create_or_extend_grad_sample(module.weight, weight_gs, batch_dim)
    bias_gs = sum_over_all_but_batch_and_last_n(B, module.bias.dim())
    create_or_extend_grad_sample(module.bias, bias_gs, batch_dim)


def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor, n_dims: int
) -> torch.Tensor:
    """
    Returns the sum of the input tensor over all dimensions except
    the first (batch) and last n_dims.

    Args:
        tensor: input tensor of shape (B, * , X[0], X[1], ..., X[n_dims-1])
        n_dims: Number of input tensor dimensions to keep

    Returns:
        New tensor of shape (B, X[0], X[1], ..., X[n_dims-1]).
        Will return the unchanged input tensor if `tensor.dim() == n_dims + 1`

    Examples:
        import torch

        A = torch.ones(2,3,4)
        print(sum_over_all_but_batch_and_last_n(A, 1))
        # prints torch.Size([2, 4])
        print(sum_over_all_but_batch_and_last_n(A, 2))
        # prints torch.Size([2, 3, 4])
    """
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)
