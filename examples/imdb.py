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
import torchtext
from opacus import PrivacyEngine
from torch.functional import F
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm


class SampleNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Embedding dimension: vocab_size + <unk>, <pad>, <eos>, <sos>
        self.emb = nn.Embedding(vocab_size + 4, 16)
        self.pool = nn.AvgPool1d(256)
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


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    for batch in tqdm(train_loader):
        data = batch.text.transpose(0, 1)
        label = batch.label

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
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            data = batch.text.transpose(0, 1)
            label = batch.label

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
        "--vocab-size",
        type=int,
        default=10_000,
        metavar="MV",
        help="Max vocab size (default: 10000)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        metavar="SL",
        help="Longer sequences will be cut to this length, shorter sequences will be padded to this length (default: 256)",
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
        "--data-root", type=str, default="../imdb", help="Where IMDB is/will be stored"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    text_field = torchtext.data.Field(
        tokenize=get_tokenizer("basic_english"),
        init_token="<sos>",
        eos_token="<eos>",
        fix_length=args.sequence_length,
        lower=True,
    )

    label_field = torchtext.data.LabelField(dtype=torch.long)

    train_data, test_data = torchtext.datasets.imdb.IMDB.splits(
        text_field, label_field, root=args.data_root
    )

    text_field.build_vocab(train_data, max_size=args.vocab_size)
    label_field.build_vocab(train_data)

    (train_iterator, test_iterator) = torchtext.data.BucketIterator.splits(
        (train_data, test_data), batch_size=args.batch_size, device=device
    )

    model = SampleNet(vocab_size=args.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            model,
            batch_size=args.batch_size,
            sample_size=len(train_data),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
        )
        privacy_engine.attach(optimizer)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_iterator, optimizer, epoch)
        evaluate(args, model, test_iterator)


if __name__ == "__main__":
    main()
