#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn


def average_gradients(model):
    world_size = torch.distributed.get_world_size()
    for param in model.parameters():
        if not param.requires_grad:
            continue
        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        param.grad /= world_size


class DifferentiallyPrivateDistributedDataParallel(nn.Module):
    def __init__(self, model):
        super().__init__()

        # Synchronize the model
        params = list(model.parameters())
        with torch.no_grad():
            for p in params:
                torch.distributed.broadcast(p.data, 0)

        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
