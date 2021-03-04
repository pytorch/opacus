#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utils for generating stats from torch tensors.
"""
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch
from torch.functional import F


def calc_sample_norms(
    named_params: Iterator[Tuple[str, torch.Tensor]], flat: bool = True
) -> List[torch.Tensor]:
    r"""
    Calculates the norm of the given tensors for each sample.

    This function calculates the overall norm of the given tensors for each sample,
    assuming the each batch's dim is zero.

    Args:
        named_params: An iterator of tuples <name, param> with name being a
            string and param being a tensor of shape ``[B, ...]`` where ``B``
            is the size of the batch and is the 0th dimension.
        flat: A flag, when set to `True` returns a flat norm over all
            layers norms

    Example:
        >>> t1 = torch.rand((2, 5))
        >>> t2 = torch.rand((2, 5))
        >>> calc_sample_norms([("1", t1), ("2", t2)])
            [tensor([1.5117, 1.0618])]

    Returns:
        A list of tensor norms where length of the list is the number of layers
    """
    norms = [param.view(len(param), -1).norm(2, dim=-1) for name, param in named_params]
    # calc norm over all layer norms if flat = True
    if flat:
        norms = [torch.stack(norms, dim=0).norm(2, dim=0)]
    return norms


def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor, n_dims: int
) -> torch.Tensor:
    r"""
    Calculates the sum over all dimensions, except the first
    (batch dimension), and excluding the last n_dims.

    This function will ignore the first dimension and it will
    not aggregate over the last n_dims dimensions.

    Args:
        tensor: An input tensor of shape ``(B, ..., X[n_dims-1])``.
        n_dims: Number of dimensions to keep.

    Example:
        >>> tensor = torch.ones(1, 2, 3, 4, 5)
        >>> sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape
        torch.Size([1, 4, 5])

    Returns:
        A tensor of shape ``(B, ..., X[n_dims-1])``
    """
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)


def unfold3d(
    tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    padding: Union[int, Tuple[int, int, int]] = 0,
    stride: Union[int, Tuple[int, int, int]] = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
):
    r"""
    Extracts sliding local blocks from an batched input tensor.

    :class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
    This method implements the same action for 5D inputs

    Args:
        tensor: An input tensor of shape ``(B, C, D, H, W)``.
        kernel_size: the size of the sliding blocks
        padding: implicit zero padding to be added on both sides of input
        stride: the stride of the sliding blocks in the input spatial dimensions
        dilation: the spacing between the kernel points.

    Example:
        >>> B, C, D, H, W = 3, 4, 5, 6, 7
        >>> tensor = torch.arange(1,B*C*D*H*W+1.).view(B,C,D,H,W)
        >>> unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
        torch.Size([3, 32, 120])

    Returns:
        A tensor of shape ``(B, C * np.product(kernel_size), L)``, where L - output spatial dimensions.
        See :class:`torch.nn.Unfold` for more details
    """

    if len(tensor.shape) != 5:
        raise ValueError(
            f"Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}"
        )

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(padding, int):
        padding = (padding, padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    if dilation != (1, 1, 1):
        raise NotImplementedError(f"dilation={dilation} not supported. We'd love a PR!")

    batch_size, channels, _, _, _ = tensor.shape

    # Input shape: (B, C, D, H, W)
    tensor = F.pad(
        tensor, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
    )
    # Output shape: (B, C, D+2*padding[2], H+2*padding[1], W+2*padding[0])

    tensor = tensor.unfold(dimension=2, size=kernel_size[0], step=stride[0])
    tensor = tensor.unfold(dimension=3, size=kernel_size[1], step=stride[1])
    tensor = tensor.unfold(dimension=4, size=kernel_size[2], step=stride[2])
    # Output shape: (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
    # For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`

    tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)
    # Output shape: (B, D_out, H_out, W_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    tensor = tensor.reshape(batch_size, -1, channels * np.prod(kernel_size)).transpose(
        1, 2
    )
    # Output shape: (B, D_out * H_out * W_out, C * kernel_size[0] * kernel_size[1] * kernel_size[2]

    return tensor
