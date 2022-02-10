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
Runs CIFAR10 training with differential privacy.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset
from torchvision import models
from tqdm import tqdm


def pretty_number(n):
    if n >= 1e6:
        return f"{n / 1e6: .2f}M"
    elif n >= 1e3:
        return f"{n / 1e3: .2f}K"
    else:
        return str(n)


def main():  # noqa: C901
    world_size = 1

    args = parse_args()
    B = args.batch_size
    H, W = args.height, args.width

    img = torch.randn(args.steps * B, 3, H, W)
    labels = torch.arange(B).repeat(args.steps)
    print(img.sum())

    train_dataset = TensorDataset(img, labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=B,
        num_workers=args.workers,
        pin_memory=True,
    )

    if not args.disable_dp:
        model = models.__dict__[args.architecture](
            pretrained=False, norm_layer=(lambda c: nn.GroupNorm(args.gn_groups, c))
        )
    else:
        model = models.__dict__[args.architecture](pretrained=False)

    model = model.to(args.device)
    print("Model size: " + pretty_number(sum([p.numel() for p in model.parameters()])))

    # Use the right distributed module wrapper if distributed training is enabled
    if world_size > 1:
        if not args.disable_dp:
            if args.dist_algo == "naive":
                model = DPDDP(model)
            elif args.dist_algo == "ddp_hook":
                model = DDP(model, device_ids=[args.device])
            else:
                raise NotImplementedError(
                    f"Unrecognized argument for the distributed algorithm: {args.dist_algo}"
                )
        else:
            model = DDP(model, device_ids=[args.device])

    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    privacy_engine = None
    if not args.disable_dp:
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                args.max_per_sample_grad_norm / np.sqrt(n_layers)
            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm

        privacy_engine = PrivacyEngine(
            secure_mode=args.secure_mode,
        )
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
        )

    criterion = nn.CrossEntropyLoss()
    model.train()
    print(type(model))

    if args.benchmark_data_loader:
        torch.cuda.synchronize()
        start = time.time()
        data_time = 0
        data_end = time.time()
        for (images, target) in tqdm(train_loader):
            data_time += time.time() - data_end
            images = images.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target.cuda(non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data_end = time.time()
    else:
        images = torch.randn(B, 3, H, W).cuda()
        target = torch.arange(B).cuda()

        torch.cuda.synchronize()
        start = time.time()
        for _ in tqdm(range(args.steps)):
            output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start
    if args.benchmark_data_loader:
        elapsed -= data_time
        print(f"Data time {data_time:.2f}")

    print(f"Took {elapsed:.2f}")
    speed = args.steps * args.batch_size / elapsed
    print(f"Speed: {speed:.2f} img/s")


def parse_args():
    parser = argparse.ArgumentParser(description="Opacus Imagenet Benchmark")
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--steps",
        default=100,
        type=int,
        help="Number of steps",
    )
    parser.add_argument(
        "--benchmark-data-loader",
        action="store_true",
        default=False,
        help="Also benchmark data loader",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size for test dataset, this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--height",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--width",
        default=224,
        type=int,
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )

    parser.add_argument(
        "--gn-groups",
        type=int,
        default=8,
        help="Number of groups in GroupNorm",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-mode",
        action="store_true",
        default=False,
        help="Enable Secure mode to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        help="path to save check points",
    )

    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device on which to run the code."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank if multi-GPU training, -1 for single GPU training. Will be overridden by the environment variables if running on a Slurm cluster.",
    )

    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
