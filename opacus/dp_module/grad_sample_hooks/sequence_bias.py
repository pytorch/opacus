#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from .utils import create_or_extend_grad_sample


def compute_grad_sample(module, A, B, batch_dim=0):
    create_or_extend_grad_sample(module.bias, B[:, -1], batch_dim)
