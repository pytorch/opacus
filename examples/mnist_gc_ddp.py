"""
Run locally: torchx run fb.dist.hpc -m .mnist_gc_ddp --img opacus_experimental_torchx -h zionex -j 1x4
Run on MAST: torchx run -s mast fb.dist.hpc -m .mnist_gc_ddp --img opacus_experimental_torchx -h zionex_80g -j 2x8

"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine

from torch.nn.parallel import DistributedDataParallel as DDP
# from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cleanup():
    dist.destroy_process_group()


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


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

    def name(self):
        return "SampleConvNet"


def train(args, model, device, train_loader, optimizer, criterion, privacy_engine, epoch):
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

    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
    if dist.get_rank() == 0:
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if dist.get_rank() == 0:
        print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def launch():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=4,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="tmp/raw_data/",
        help="Where MNIST is/will be stored",
    )
    args = parser.parse_args()
    args.data_root = os.path.join(args.data_root, str(os.getpid()))
    train_dataset = datasets.MNIST(args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        sampler = DistributedSampler(train_dataset),
    )
    test_loader = DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = []
    # device = torch.device(f"cuda:{rank}")
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    print(f"Using device: {device}")
    model = DDP(SampleConvNet().to(device), device_ids=[local_rank])
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
    privacy_engine = PrivacyEngine()
    model, optimizer, criterion, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            poisson_sampling=False,
            grad_sample_mode="ghost"
        )
    print(f"(rank {local_rank}) Average batch size per GPU: {int(optimizer.expected_batch_size)}")

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, privacy_engine, epoch)
    run_results.append(test(model, device, test_loader))
    cleanup()


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = dist.get_rank()
    print(
        f"Is cuda available: {torch.cuda.is_available()}, device: {local_rank}, rank: {rank}"
    )
    launch()

if __name__ == "__main__":
    main()
