#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.functional import F

from .utils import create_or_extend_grad_sample


def compute_grad_sample(module, A, B, batch_dim=0):
    one_hot = F.one_hot(A, num_classes=module.weight.shape[0])
    gs = torch.einsum("n...i,n...j->n...ij", one_hot, B)

    create_or_extend_grad_sample(
        module.weight, torch.einsum("n...ij->nij", gs), batch_dim
    )
