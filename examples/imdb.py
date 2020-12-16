#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Training sentiment prediction model on IMDB movie reviews dataset.
Architecture and reference results from https://arxiv.org/pdf/1911.11607.pdf
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from opacus import PrivacyEngine
from torch.functional import F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast


class SampleNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def name(self):
        return "SampleNet"


def binary_accuracy(preds, y):
    correct = (y.long() == torch.argmax(preds, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y


def train(args, model, train_loader, optimizer, epoch):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.train().to(device)

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predictions = model(data).squeeze(1)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Train Loss: {np.mean(losses):.6f} "
            f"Train Accuracy: {np.mean(accuracies):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(
            f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} ] \t Accuracy: {np.mean(accuracies):.6f}"
        )


def evaluate(args, model, test_loader):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    device = torch.device(args.device)
    model = model.eval().to(device)

    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)

            predictions = model(data).squeeze(1)

            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            losses.append(loss.item())
            accuracies.append(acc.item())

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            np.mean(losses), np.mean(accuracies) * 100
        )
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch IMDB Example")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        metavar="LR",
        help="learning rate (default: .02)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.56,
        metavar="S",
        help="Noise multiplier (default 0.56)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        metavar="SL",
        help="Longer sequences will be cut to this length (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla optimizer",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root", type=str, default="../imdb", help="Where IMDB is/will be stored"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    raw_dataset = load_dataset("imdb", cache_dir="imdb")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=args.max_sequence_length
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if args.secure_rng:
        try:
            import torchcsprng as prng
        except ImportError as e:
            msg = (
                "To use secure RNG, you must install the torchcsprng package! "
                "Check out the instructions here: https://github.com/pytorch/csprng#installation"
            )
            raise ImportError(msg) from e

        generator = prng.create_random_device_generator("/dev/urandom")

    else:
        generator = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        generator=generator,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    model = SampleNet(vocab_size=len(tokenizer)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            model,
            batch_size=args.batch_size,
            sample_size=len(train_dataset),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            secure_rng=args.secure_rng,
        )
        privacy_engine.attach(optimizer)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        evaluate(args, model, test_loader)


if __name__ == "__main__":
    main()
