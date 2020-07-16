#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
utils for generating stats from torch tensors.
"""
from typing import Dict, Iterator, List, Tuple

import torch


def calc_sample_norms(
    named_params: Iterator[Tuple[str, torch.Tensor]], flat: bool = True
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Calculates the (overall) norm of the given tensors over each sample,
    assuming dim=0 is represnting the sample in the batch.

    Returns:
        A tuple with first element being a list of torch tensors all of size
        B (look at `named_params`). Each element in the list corresponds to
        the norms of the parameter appearing in the same order of the
        `named_params`.

    Arguments:
        named_params: An iterator of tuples each representing a named tensor with
            name being a string and param a tensor of shape [B, XYZ...] where B
            is the size of the batch and is the 0th dimension
        flat: a flag, when set to `True` returns a flat norm over all
            layers, i.e. norm of all the norms across layers for each sample.
        stats_required: a flag, when set to True, the function will provide some
            statistics over the batch, including mean, median, and max values
    """
    norms = [param.view(len(param), -1).norm(2, dim=-1) for name, param in named_params]
    # calc norm over all layer norms if flat = True
    if flat:
        norms = [torch.stack(norms, dim=0).norm(2, dim=0)]
    return norms


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
