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

import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModuleFastGradientClippingTP
from opacus.optimizers import FSDPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleModule(nn.Module):
    def __init__(self):
        super(SampleModule, self).__init__()
        self.fc1 = nn.Linear(4, 32, bias=False)
        self.fc2 = nn.Linear(32, 4, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).flatten(start_dim=1)
        x = F.softmax(x)
        return x


# pyre-ignore
def model_parallel(rank, world_size, m):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print("current_local_rank is", rank)

    torch.cuda.set_device(rank)

    input = torch.tensor([(-1.0, -1.0, -1.0, -1.0), (-2.0, -2.0, -2.0, -2.0)]).to(
        device="cuda"
    )
    input = torch.unsqueeze(input, 0)
    print("input shape is", input.shape)

    tp_mesh = init_device_mesh("cuda", (2,))
    sharded_model = parallelize_module(
        m, tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    )

    DP_sharded_model = GradSampleModuleFastGradientClippingTP(
        sharded_model, loss_reduction="mean", batch_first=True
    )

    optimizer_gc = torch.optim.SGD(DP_sharded_model.parameters(), lr=1)
    optimizer_gc = FSDPOptimizerFastGradientClipping(
        optimizer_gc,
        noise_multiplier=0.0,
        max_grad_norm=0.1,
        expected_batch_size=1,
        loss_reduction="mean",
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    criterion_gc = DPLossFastGradientClipping(
        DP_sharded_model, optimizer_gc, copy.deepcopy(criterion)
    )
    optimizer_gc.zero_grad()

    output = DP_sharded_model(input)
    print("output shape is", output.shape)
    label = torch.tensor([3]).to(device="cuda")

    loss = criterion_gc(output, label)
    loss.backward()
    optimizer_gc.step()

    for name, param in DP_sharded_model.named_parameters():
        print(name, param.grad)


def main():
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    m = SampleModule()
    mp.spawn(
        model_parallel,
        args=(world_size, copy.deepcopy(m)),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
