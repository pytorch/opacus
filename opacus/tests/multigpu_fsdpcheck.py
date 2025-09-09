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


import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.fsdp_utils import FSDP2Wrapper
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


PRIVACY_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def setup(rank, world_size):
    if sys.platform == "win32":
        raise ValueError("Windows platform is not supported for this test")
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)
        self.net3 = nn.LayerNorm(5)
        self.net4 = nn.Linear(5, 5)

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.relu(self.net1(x)))))


def demo_basic(rank, weight, world_size, grad_sample_mode, mixed_precision):
    torch.manual_seed(world_size)
    batch_size = 32
    torch.cuda.set_device(rank)
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model.net1.weight.data.zero_()

    # create dataset
    labels = torch.randn(2 * batch_size, 5).to(rank)
    data = torch.randn(2 * batch_size, 10)
    dataset = TensorDataset(data, labels)
    max_grad_norm = 1
    # we set the seed to be same for all workers, so the noise generated on rank 0 for DP-DDP should match the noise generated on all the workers for FSDP
    noise_multiplier = 5.0

    if grad_sample_mode == "ghost":
        dp_model = DPDDP(model)
    else:
        if not mixed_precision:
            dp_model = FSDP2Wrapper(model)
        else:
            dp_model = FSDP2Wrapper(
                model,
                mp_policy=dist.fsdp.MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
                opacus_high_precision_layers=(nn.LayerNorm,),
            )

    optimizer = optim.SGD(model.parameters(), lr=1)

    privacy_engine = PrivacyEngine()

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    dp_model, optimizer, loss_fn, data_loader = privacy_engine.make_private(
        module=dp_model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        grad_sample_mode=grad_sample_mode,
    )
    if grad_sample_mode == "ghost" and mixed_precision is True:
        for x, y in data_loader:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = dp_model(x.to(rank))
                assert outputs.dtype == torch.bfloat16
                loss = loss_fn(outputs, y)
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
    else:
        for x, y in data_loader:
            outputs = dp_model(x.to(rank))
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            if mixed_precision is True:
                assert outputs.dtype == torch.bfloat16
            optimizer.step()
            break
    if grad_sample_mode == "ghost_fsdp":
        full_weight = model.net1.weight.full_tensor()
    else:
        full_weight = model.net1.weight
    weight.copy_(full_weight.data.cpu())
    cleanup()


def run_demo(demo_fn, weight, world_size, grad_sample_mode, mixed_precision):
    mp.spawn(
        demo_fn,
        args=(weight, world_size, grad_sample_mode, mixed_precision),
        nprocs=world_size,
        join=True,
    )


class GradientComputationTestFSDP(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_gradient_correct_fsdp(self) -> None:
        # Tests that gradient is the same with DDP or with FSDP

        weight_ddp, weight_fsdp = torch.zeros(10, 10), torch.zeros(10, 10)

        run_demo(
            demo_basic,
            weight_ddp,
            2,
            grad_sample_mode="ghost",
            mixed_precision=False,
        )
        run_demo(
            demo_basic,
            weight_fsdp,
            2,
            grad_sample_mode="ghost_fsdp",
            mixed_precision=False,
        )
        self.assertTrue(
            not torch.allclose(weight_fsdp, torch.zeros(10, 10), atol=1e-5, rtol=1e-3)
        )
        self.assertTrue(torch.allclose(weight_ddp, weight_fsdp, atol=1e-5, rtol=1e-3))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_gradient_correct_fsdp_mixed_precision(self) -> None:
        # Tests that gradient is the same with DDP or with FSDP under mixed precision

        weight_ddp, weight_fsdp = torch.zeros(10, 10), torch.zeros(10, 10)

        run_demo(
            demo_basic,
            weight_ddp,
            2,
            grad_sample_mode="ghost",
            mixed_precision=True,
        )
        run_demo(
            demo_basic,
            weight_fsdp,
            2,
            grad_sample_mode="ghost_fsdp",
            mixed_precision=True,
        )
        self.assertTrue(
            not torch.allclose(weight_fsdp, torch.zeros(10, 10), atol=1e-5, rtol=1e-3)
        )
        self.assertTrue(torch.allclose(weight_ddp, weight_fsdp, atol=1e-5, rtol=1e-3))
