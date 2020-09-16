#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.functional import F

from .utils import create_or_extend_grad_sample


def compute_grad_sample(module, A, B, batch_dim=0):
    gs = F.group_norm(A, module.num_groups, eps=module.eps) * B
    create_or_extend_grad_sample(
        module.weight, torch.einsum("ni...->ni", gs), batch_dim
    )
    if module.bias is not None:
        create_or_extend_grad_sample(
            module.bias, torch.einsum("ni...->ni", B), batch_dim
        )
