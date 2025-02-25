#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

#  buck2 run @//mode/opt-split-dwarf :tests

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

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, weight, world_size, dpddp, clipping, grad_sample_mode):
    torch.manual_seed(world_size)
    batch_size = 64
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model.net1.weight.data.zero_()
    optimizer = optim.SGD(model.parameters(), lr=1)

    labels = torch.randn(2 * batch_size, 5).to(rank)
    data = torch.randn(2 * batch_size, 10)

    dataset = TensorDataset(data, labels)

    loss_fn = nn.MSELoss()
    if dpddp:
        ddp_model_v0 = DPDDP(model)
    else:
        ddp_model_v0 = DDP(model, device_ids=[rank])
    print("dpddp = ", dpddp, "clipping = ", clipping, "grad_sample_mode = ", grad_sample_mode)

    privacy_engine = PrivacyEngine()

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    max_grad_norm = 1e8
    if clipping == "per_layer":
        max_grad_norm = [max_grad_norm for _ in model.parameters()]
    ddp_model, optimizer, criterion, data_loader = privacy_engine.make_private(
        module=ddp_model_v0,
        optimizer=optimizer,
        criterion=loss_fn,
        data_loader=data_loader,
        noise_multiplier=0,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
        clipping=clipping,
        grad_sample_mode=grad_sample_mode,
    )

    for x, y in data_loader:
        outputs = ddp_model(x.to(rank))
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

    weight.copy_(model.net1.weight.data.cpu())
    cleanup()


def run_demo(demo_fn, weight, world_size, dpddp, clipping, grad_sample_mode):
    # mp.spawn(
    #     demo_fn,
    #     args=(weight, world_size, dpddp, clipping, grad_sample_mode),
    #     nprocs=world_size,
    #     join=True,
    # )
    demo_fn(0, weight, world_size, dpddp, clipping, grad_sample_mode)


def main():
    n_gpus = torch.cuda.device_count()
    assert(
        n_gpus >= 2, f"Need at least 2 gpus but was provided only {n_gpus}."
    )

    for clipping in ["flat", "per_layer"]:
        for grad_sample_mode in ["ghost"]:
            weight_dpddp, weight_ddp = torch.zeros(10, 10), torch.zeros(10, 10)
            print("starting tests")
            run_demo(
                demo_basic,
                weight_dpddp,
                1,
                dpddp=False,
                clipping=clipping,
                grad_sample_mode=grad_sample_mode,
            )
            print("Ran DDP")
            run_demo(
                demo_basic,
                weight_ddp,
                1,
                dpddp=True,
                clipping=clipping,
                grad_sample_mode=grad_sample_mode,
            )

            assert(
                torch.allclose(weight_ddp, weight_dpddp, atol=1e-5, rtol=1e-3)
            )
            print("Test passed for clipping = ", clipping, " and grad_sample_mode = ", grad_sample_mode)
        print("All test passed")
