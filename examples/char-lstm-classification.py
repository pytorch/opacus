#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.layers import DPGRU, DPLSTM, DPRNN
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="PyTorch Name language classification DP Training"
)
parser.add_argument(
    "--data-root", type=str, help="Path to training set of names (ie. ~/data/names/)"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="GPU ID for this process (default: 'cuda')",
)
parser.add_argument(
    "-b",
    "--batch-size-test",
    default=1600,
    type=int,
    metavar="N",
    help="mini-batch size for test dataset (default: 1600)",
)
parser.add_argument(
    "-sr",
    "--sample-rate",
    default=0.05,
    type=float,
    metavar="SR",
    help="sample rate used for batch construction (default: 0.05)",
)
parser.add_argument(
    "--mode",
    default="lstm",
    choices=["lstm", "gru", "rnn"],
    help="recursive network type",
)
parser.add_argument(
    "--embedding-size", default=64, type=int, help="Character embedding dimension"
)
parser.add_argument(
    "--hidden-size", default=128, type=int, help="hidden state dimensions"
)
parser.add_argument("--n-layers", default=1, type=int, help="How many layers to use")
parser.add_argument(
    "--test-every",
    default=0,
    type=int,
    help="Run evaluation on the test every these many epochs",
)
parser.add_argument(
    "--bidirectional",
    action="store_true",
    default=False,
    help="If turned on, makes the RNN bidirectional",
)
parser.add_argument(
    "--learning-rate",
    default=2.0,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument(
    "--train-split",
    type=float,
    default=0.8,
    help="Fraction of data to utilize for training (rest for evaluation)",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    metavar="S",
    help="Noise multiplier (default 1.0)",
)
parser.add_argument(
    "-c",
    "--max-per-sample-grad-norm",
    type=float,
    default=1.5,
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
    "--secure-rng",
    action="store_true",
    default=False,
    help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
)
parser.add_argument(
    "--delta",
    type=float,
    default=8e-5,
    metavar="D",
    help="Target delta (default: 1e-5)",
)
parser.add_argument(
    "--print-every",
    type=int,
    default=5,
    help="Print the evaluation accuracy every these many iterations",
)


class CharByteEncoder(nn.Module):
    """
    This encoder takes a UTF-8 string and encodes its bytes into a Tensor. It can also
    perform the opposite operation to check a result.

    Examples:

    >>> encoder = CharByteEncoder()
    >>> t = encoder('Ślusàrski')  # returns tensor([256, 197, 154, 108, 117, 115, 195, 160, 114, 115, 107, 105, 257])
    >>> encoder.decode(t)  # returns "<s>Ślusàrski</s>"
    """

    def __init__(self):
        super().__init__()
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.pad_token = "<pad>"

        self.start_idx = 256
        self.end_idx = 257
        self.pad_idx = 258

    def forward(self, s: str, pad_to=0) -> torch.LongTensor:
        """
        Encodes a string. It will append a start token <s> (id=self.start_idx) and an end token </s>
        (id=self.end_idx).

        Args:
            s: The string to encode.
            pad_to: If not zero, pad by appending self.pad_idx until string is of length `pad_to`.
                Defaults to 0.

        Returns:
            The encoded LongTensor of indices.
        """
        encoded = s.encode()
        n_pad = pad_to - len(encoded) if pad_to > len(encoded) else 0
        return torch.LongTensor(
            [self.start_idx]
            + [c for c in encoded]  # noqa
            + [self.end_idx]
            + [self.pad_idx for _ in range(n_pad)]
        )

    def decode(self, char_ids_tensor: torch.LongTensor) -> str:
        """
        The inverse of `forward`. Keeps the start, end and pad indices.
        """
        char_ids = char_ids_tensor.cpu().detach().tolist()

        out = []
        buf = []
        for c in char_ids:
            if c < 256:
                buf.append(c)
            else:
                if buf:
                    out.append(bytes(buf).decode())
                    buf = []
                if c == self.start_idx:
                    out.append(self.start_token)
                elif c == self.end_idx:
                    out.append(self.end_token)
                elif c == self.pad_idx:
                    out.append(self.pad_token)

        if buf:  # in case some are left
            out.append(bytes(buf).decode())
        return "".join(out)

    def __len__(self):
        """
        The length of our encoder space. This is fixed to 256 (one byte) + 3 special chars
        (start, end, pad).

        Returns:
            259
        """
        return 259


class NamesDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)

        self.labels = list({langfile.stem for langfile in self.root.iterdir()})
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        self.encoder = CharByteEncoder()
        self.samples = self.construct_samples()

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

    def construct_samples(self):
        samples = []
        for langfile in self.root.iterdir():
            label_name = langfile.stem
            label_id = self.labels_dict[label_name]
            with open(langfile, "r") as fin:
                for row in fin:
                    samples.append(
                        (self.encoder(row.strip()), torch.tensor(label_id).long())
                    )
        return samples

    def label_count(self):
        cnt = Counter()
        for _x, y in self.samples:
            label = self.labels[int(y)]
            cnt[label] += 1
        return cnt


VOCAB_SIZE = 256 + 3  # 256 alternatives in one byte, plus 3 special characters.


class CharNNClassifier(nn.Module):
    def __init__(
        self,
        rnn_type,
        embedding_size,
        hidden_size,
        output_size,
        num_layers=1,
        bidirectional=False,
        vocab_size=VOCAB_SIZE,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = rnn_type(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # -> [B, T, D]
        x, _ = self.rnn(x, hidden)  # -> [B, T, H]
        x = x[:, -1, :]  # -> [B, H]
        x = self.out_layer(x)  # -> [B, C]
        return x


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem[0] for elem in batch], batch_first=True, padding_value=padding_idx
    )
    y = torch.stack([elem[1] for elem in batch]).long()

    return x, y


def train(model, criterion, optimizer, train_loader, epoch, device="cuda:0"):
    model.train()

    accs = []
    losses = []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        preds = logits.argmax(-1)
        n_correct = float(preds.eq(y).sum())
        batch_accuracy = n_correct / len(y)

        accs.append(batch_accuracy)
        losses.append(float(loss))

    printstr = (
        f"\t Epoch {epoch}. Accuracy: {mean(accs):.6f} | Loss: {mean(losses):.6f}"
    )
    try:
        privacy_engine = optimizer.privacy_engine
        epsilon, best_alpha = privacy_engine.get_privacy_spent()
        printstr += f" | (ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
    except AttributeError:
        pass
    print(printstr)
    return


def test(model, test_loader, privacy_engine, device="cuda:0"):
    model.eval()

    accs = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)

            preds = model(x).argmax(-1)
            n_correct = float(preds.eq(y).sum())
            batch_accuracy = n_correct / len(y)

            accs.append(batch_accuracy)
    mean_acc = mean(accs)
    printstr = "\n----------------------------\n" f"Test Accuracy: {mean_acc:.6f}"
    if privacy_engine:
        epsilon, best_alpha = privacy_engine.get_privacy_spent()
        printstr += f" (ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}"
    print(printstr + "\n----------------------------\n")
    return mean_acc


def main():
    args = parser.parse_args()
    device = torch.device(args.device)
    ds = NamesDataset(args.data_root)
    train_len = int(args.train_split * len(ds))
    test_len = len(ds) - train_len

    print(f"{train_len} samples for training, {test_len} for testing")

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

    train_ds, test_ds = torch.utils.data.random_split(
        ds, [train_len, test_len], generator=generator
    )

    if args.mode == "rnn":
        rnn_type = DPRNN
    elif args.mode == "gru":
        rnn_type = DPGRU
    elif args.mode == "lstm":
        rnn_type = DPLSTM
    else:
        raise ValueError(f"Invalid network type: {args.mode}")
    model = CharNNClassifier(
        rnn_type,
        args.embedding_size,
        args.hidden_size,
        len(ds.labels),
        args.n_layers,
        args.bidirectional,
    )
    model = model.to(device)

    train_ds, test_ds = torch.utils.data.random_split(
        ds, [train_len, test_len], generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        num_workers=1,
        pin_memory=True,
        generator=generator,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_ds), sample_rate=args.sample_rate, generator=generator
        ),
        collate_fn=padded_collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=padded_collate,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=args.sample_rate,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            target_delta=args.delta,
            secure_rng=args.secure_rng,
        )
        privacy_engine.attach(optimizer)
    else:
        privacy_engine = None

    print(f"Train stats ({args.mode}): \n")
    for epoch in tqdm(range(args.epochs)):
        train(model, criterion, optimizer, train_loader, epoch, device=device)
        if args.test_every:
            if epoch % args.test_every == 0:
                test(model, test_loader, privacy_engine, device=device)

    mean_acc = test(model, test_loader, privacy_engine, device=device)
    torch.save(mean_acc, f"run_results_chr_{args.mode}_classification.pt")


if __name__ == "__main__":
    main()
