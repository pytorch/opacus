#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from .utils import create_or_extend_grad_sample


def compute_grad_sample(module, A, B, batch_dim=0):
    n = A.shape[0]
    A = torch.nn.functional.unfold(
        A, module.kernel_size, padding=module.padding, stride=module.stride
    )
    B = B.reshape(n, -1, A.shape[-1])
    try:
        # n=batch_sz; o=num_out_channels; p=num_in_channels*kernel_sz
        grad_sample = (
            torch.einsum("noq,npq->nop", B, A)
            if module.groups == 1
            else torch.einsum("njk,njk->nj", B, A)
        )
        shape = [n] + list(module.weight.shape)
        create_or_extend_grad_sample(
            module.weight, grad_sample.reshape(shape), batch_dim
        )
    except Exception as e:
        raise type(e)(
            f"{e} There is probably a problem with nn.Conv2D.groups"
            + "It should be either 1 or in_channel"
        )
    if module.bias is not None:
        create_or_extend_grad_sample(module.bias, torch.sum(B, dim=2), batch_dim)
