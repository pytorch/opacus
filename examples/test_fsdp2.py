#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

#  command to run: buck2 run @//mode/opt -c hpc_comms.use_nccl=stable :test_fsdp2

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed._composable.fsdp import fully_shard
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    if sys.platform == "win32":
        raise ValueError("Windows platform is not supported for this test")
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12855"

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

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, weight, world_size, clipping, grad_sample_mode):
    torch.manual_seed(world_size)
    batch_size = 8
    torch.cuda.set_device(rank)
    setup(rank, world_size)

    labels = torch.randint(0, 5, (world_size * batch_size,)).to(rank)
    data = torch.randn(world_size * batch_size, 10)

    dataset = TensorDataset(data, labels)

    criterion = nn.CrossEntropyLoss()

    print("clipping = ", clipping, "grad_sample_mode = ", grad_sample_mode)

    privacy_engine = PrivacyEngine()

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    max_grad_norm = 1e8
    if grad_sample_mode == "ghost":
        model = ToyModel().to(rank)
        model.net1.weight.data.zero_()
        optimizer = optim.SGD(model.parameters(), lr=1)
        ddp_model = DPDDP(model)
        final_model, optimizer, criterion, data_loader = privacy_engine.make_private(
            module=ddp_model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=0,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
        )
    elif grad_sample_mode == "ghost_fsdp":
        # create model and move it to GPU with id rank
        model = ToyModel()
        model.net1.weight.data.zero_()
        for module in model.children():
            fully_shard(module)
        fsdp_model = fully_shard(model)
        optimizer = optim.SGD(model.parameters(), lr=1)

        final_model, optimizer, criterion, data_loader = privacy_engine.make_private(
            module=fsdp_model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=data_loader,
            noise_multiplier=0,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
        )
    else:
        model = ToyModel().to(rank)
        final_model = DDP(model, device_ids=[rank])

    for x, y in data_loader:
        outputs = final_model(x.to(rank))
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break
    if grad_sample_mode == "ghost_fsdp":
        full_weight = model.net1.weight.full_tensor()
    else:
        full_weight = model.net1.weight
    weight.copy_(full_weight.data.cpu())
    cleanup()


def run_demo(demo_fn, weight, world_size, clipping, grad_sample_mode):
    mp.spawn(
        demo_fn,
        args=(weight, world_size, clipping, grad_sample_mode),
        nprocs=world_size,
        join=True,
    )


def main():
    n_gpus = torch.cuda.device_count()
    world_size = 4
    assert (
        n_gpus >= world_size,
        f"Need at least {world_size} gpus but was provided only {n_gpus}.",
    )

    for clipping in ["flat"]:
        weight_gc_fsdp, weight_gc_ddp = torch.zeros(10, 10), torch.zeros(10, 10)
        print("starting tests")
        run_demo(
            demo_basic,
            weight_gc_fsdp,
            world_size,
            clipping=clipping,
            grad_sample_mode="ghost_fsdp",
        )
        run_demo(
            demo_basic,
            weight_gc_ddp,
            world_size,
            clipping=clipping,
            grad_sample_mode="ghost",
        )

        print("Ran all the tests for cliiping = ", clipping)
        assert torch.allclose(weight_gc_fsdp, weight_gc_ddp, atol=1e-8, rtol=1e-5)
        print("Test passed for clipping = ", clipping)
    print("All tests passed")
