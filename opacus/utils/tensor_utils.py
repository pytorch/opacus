#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utils for generating stats from torch tensors.
"""
from typing import Iterator, List, Tuple

import torch


def calc_sample_norms(
    named_params: Iterator[Tuple[str, torch.Tensor]], flat: bool = True
) -> List[torch.Tensor]:
    r"""
    Calculates the norm of the given tensors for each sample.

    This function calculates the overall norm of the given tensors for each sample,
    assuming the each batch's dim is zero.

    Parameters
    ----------
    named_params: Iterator[Tuple[str, torch.Tensor]]
        An iterator of tuples <name, param> with name being a string
        and param being a tensor of shape ``[B, ...]`` where ``B``
        is the size of the batch and is the 0th dimension.
    flat: bool
        A flag, when set to `True` returns a flat norm over all
        layers norms

    Example
    -------
        >>> t1 = torch.rand((2, 5))
        >>> t2 = torch.rand((2, 5))
        >>> calc_sample_norms([("1", t1), ("2", t2)])
            [tensor([1.5117, 1.0618])]

    Returns
    -------
        List[torch.Tensor]
            A list of tensor norms where length of the list is the number of layers
    """
    norms = [param.view(len(param), -1).norm(2, dim=-1) for name, param in named_params]
    # calc norm over all layer norms if flat = True
    if flat:
        # pyre-fixme[6]: Expected `Union[List[torch.Tensor],
        #  typing.Tuple[torch.Tensor, ...]]` for 1st param but got
        #  `List[torch.FloatTensor]`.
        norms = [torch.stack(norms, dim=0).norm(2, dim=0)]
    # pyre-fixme[7]: Expected `Tuple[List[torch.Tensor], Dict[str, float]]` but got
    #  `List[torch.FloatTensor]`.
    return norms


def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor, n_dims: int
) -> torch.Tensor:
    r"""
    Calculates the sum over all dimensions, except the first (batch dimension), and excluding the last n_dims.

    This function will ignore the first dimension and it will not aggregate over the last n_dims dimensions.

    Parameters
    ----------
    tensor: torch.Tensor
        An input tensor of shape ``(B, ..., X[n_dims-1])``.
    n_dims: int
        Number of dimensions to keep.

    Example
    -------
        >>> tensor = torch.ones(1, 2, 3, 4, 5)
        >>> sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape
        torch.Size([1, 4, 5])

    Returns
    -------
        torch.Tensor
            A tensor of shape ``(B, ..., X[n_dims-1])``
    """
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)
