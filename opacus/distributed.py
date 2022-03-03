#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


def average_gradients(model: nn.Module) -> None:
    """
    For all parameters of a given ``model`` averages gradients over all workers

    Args:
        model: model

    Returns:
        None
    """
    world_size = torch.distributed.get_world_size()
    for param in model.parameters():
        if not param.requires_grad:
            continue
        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        param.grad /= world_size


class DifferentiallyPrivateDistributedDataParallel(nn.Module):
    """
    Implements distributed data parallelism that is based on
    ``torch.distributed`` package at the module level.

    """

    def __init__(self, model: nn.Module):
        super().__init__()

        # Synchronize the model
        params = list(model.parameters())
        with torch.no_grad():
            for p in params:
                torch.distributed.broadcast(p.data, 0)

        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
