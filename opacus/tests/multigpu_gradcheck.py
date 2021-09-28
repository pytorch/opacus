#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.layers import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP


PRIVACY_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def setup(rank, world_size):
    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo", init_method=init_method, rank=rank, world_size=world_size
        )
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        # dist.init_process_group("gloo", rank=rank, world_size=world_size)

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

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, weight, world_size, dp):
    torch.manual_seed(world_size)
    batch_size = 32
    withdp = "with" + ("out " if not dp else "")
    print(f"Running basic DDP {withdp} differential privacy example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    if dp:
        ddp_model = DPDDP(model)
        engine = PrivacyEngine(
            ddp_model,
            batch_size=batch_size,
            sample_size=10 * batch_size,
            alphas=PRIVACY_ALPHAS,
            noise_multiplier=0,
            max_grad_norm=1e8,
        )
    else:
        ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1)
    if dp:
        engine.attach(optimizer)

    # if rank == 0:
    #     print(model.net1.weight)
    optimizer.zero_grad()
    labels = torch.randn(batch_size, 5).to(rank)
    outputs = ddp_model(torch.randn(batch_size, 10).to(rank))
    loss_fn(outputs, labels).backward()
    optimizer.step()
    # if rank == 0:
    #     print(model.net1.weight)

    weight.copy_(model.net1.weight.data.cpu())

    cleanup()


def run_demo(demo_fn, weight, world_size, dp):
    mp.spawn(demo_fn, args=(weight, world_size, dp), nprocs=world_size, join=True)


class GradientComputationTest(unittest.TestCase):
    def test_gradient_correct(self):
        # Tests that gradient is the same with DP or with DDP
        n_gpus = torch.cuda.device_count()
        self.assertTrue(
            n_gpus >= 2, f"Need at least 2 gpus but was provided only {n_gpus}."
        )
        weight_dp, weight_nodp = torch.zeros(10, 10), torch.zeros(10, 10)
        run_demo(demo_basic, weight_dp, 2, dp=True)
        run_demo(demo_basic, weight_nodp, 2, dp=False)

        self.assertTrue(torch.norm(weight_dp - weight_nodp) < 1e-7)
