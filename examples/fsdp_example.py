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

"""
Runs MNIST training with Ghost Clipping DP-SGD and FSDP2.

"""

import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.fsdp_utils import FSDP2Wrapper
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

DATA_ROOT = "./mnist"

mnist_train_ds = datasets.MNIST(
    DATA_ROOT,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    ),
)
mnist_test_ds = datasets.MNIST(
    DATA_ROOT,
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    ),
)

LR = 0.1
BATCH_SIZE = 256
N_GPUS = torch.cuda.device_count()


# pyre-ignore
def init_training(rank):
    model = SampleConvNet()
    # Similar to DPDDP, FSDP2 requires the model to be wrapped with the corresponding wrapper
    model = FSDP2Wrapper(model)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0)
    data_loader = DataLoader(
        mnist_train_ds,
        # batch_size=BATCH_SIZE // N_GPUS, -- non-private or with poisson_sampling = False
        batch_size=BATCH_SIZE,
        # sampler=DistributedSampler(mnist_train_ds), -- non-private or with poisson_sampling = False
        num_workers=0,
        pin_memory=True,
    )

    if rank == 0:
        logger.info(
            f"(rank {rank}) Initialized model ({type(model).__name__}), "
            f"optimizer ({type(optimizer).__name__}), "
            f"data loader ({type(data_loader).__name__}, len={len(data_loader)})"
        )

    privacy_engine = PrivacyEngine()

    # PrivacyEngine looks at the model's class and enables
    # distributed processing if it's wrapped with DPDDP or FSDP
    # pyre-ignore
    model, optimizer, criterion, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        data_loader=data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        grad_sample_mode="ghost_fsdp",  # ghost clipping with FSDP2
        poisson_sampling=True,
    )

    if rank == 0:
        logger.info(
            f"(rank {rank}) After privatization: model ({type(model).__name__}), "
            f"optimizer ({type(optimizer).__name__}), "
            f"data loader ({type(data_loader).__name__}, len={len(data_loader)})"
        )

    logger.info(
        f"(rank {rank}) Average batch size per GPU: {int(optimizer.expected_batch_size)}"
    )

    return model, optimizer, criterion, data_loader, privacy_engine


# pyre-ignore
def test(model, device):
    test_loader = DataLoader(
        mnist_test_ds,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    model.train()
    return correct / len(mnist_test_ds)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def launch(rank, world_size, epochs):
    # set the device for the current process
    torch.cuda.set_device(rank)
    # setup environment for distributed training
    setup(rank, world_size)

    model, optimizer, criterion, train_loader, privacy_engine = init_training(rank)
    model.train()

    for epoch in range(1, epochs + 1):
        train_loss = train(model, rank, train_loader, optimizer, criterion, epoch)

        test_accuracy = test(model, rank)
        epsilon = privacy_engine.get_epsilon(delta=1e-5)

        if rank == 0:
            print(
                f"Epoch: {epoch} \t"
                f"Train Loss: {train_loss:.3f} \t"
                f"Test Accuracy: {test_accuracy:.2f} |"
                f"(ε = {epsilon:.2f})"
            )

    cleanup()


def main():
    EPOCHS = 10
    world_size = torch.cuda.device_count()
    mp.spawn(
        launch,
        args=(
            world_size,
            EPOCHS,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
