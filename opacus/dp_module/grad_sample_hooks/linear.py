#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from .utils import create_or_extend_grad_sample


def compute_grad_sample(module, A, B, batch_dim=0):
    gs = torch.einsum("n...i,n...j->n...ij", B, A)
    create_or_extend_grad_sample(
        module.weight, torch.einsum("n...ij->nij", gs), batch_dim
    )
    if module.bias is not None:
        create_or_extend_grad_sample(
            module.bias, torch.einsum("n...k->nk", B), batch_dim
        )
