"""
Runs Adult training with differential privacy.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchdp import PrivacyEngine
from tqdm import tqdm
from compute_gdp_privacy import *


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(123, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        #        x = x.view(-1, 123)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "SampleConvNet"


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(f"Train Epoch: {epoch} \t" f"Loss: {np.mean(losses):.6f} ")
        print(
            "Moments Accountant gives: "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha:.2f}"
        )
        mu = compute_muP(epoch, args.sigma, 60000, args.batch_size)
        epsi = compute_epsP(epoch, args.sigma, 60000, args.batch_size, args.delta)
        print(
            "GDP Central Limit Theorem gives: "
            f"(μ = {mu:.2f}, ε = {epsi:.2f}, δ = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(args, model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target.long()).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


# custom dataset
class AdultDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.X = images
        self.y = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        return (data.to_numpy(), self.y[i])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Adult Example")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.15,
        metavar="LR",
        help="learning rate (default: .1)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.55,
        metavar="S",
        help="Noise multiplier (default 1.0)",
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
        help="Disable privacy training and just train with vanilla SGD",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    kwargs = {"num_workers": 1, "pin_memory": True}

    """Loads ADULT a2a as in LIBSVM and preprocesses to combine training and validation data."""
    # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    x = pd.read_csv("adult.csv")
    trainData, testData = train_test_split(x, test_size=0.1, random_state=218)
    # have to reset index, see https://discuss.pytorch.org/t/keyerror-when-enumerating-over-dataloader/54210/13
    trainData = trainData.reset_index()
    testData = testData.reset_index()

    train_data = trainData.iloc[:, 1:-1].astype("float32")
    test_data = testData.iloc[:, 1:-1].astype("float32")
    train_labels = (trainData.iloc[:, -1] == 1).astype("int32")
    test_labels = (testData.iloc[:, -1] == 1).astype("int32")

    train_loader = torch.utils.data.DataLoader(
        AdultDataset(train_data, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        AdultDataset(test_data, test_labels),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    run_results = []
    for _ in range(args.n_runs):
        model = SampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        if not args.disable_dp:
            privacy_engine = PrivacyEngine(
                model,
                batch_size=args.batch_size,
                sample_size=len(train_loader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)]
                + list(np.arange(12, 60, 0.1))
                + list(np.arange(61, 100, 1)),
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )
            privacy_engine.attach(optimizer)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

        run_results.append(test(args, model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )


if __name__ == "__main__":
    main()
