#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs MNIST training with differential privacy.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.cli import LightningCLI

from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torchvision import datasets, transforms
import pytorch_lightning as pl
import torchmetrics


import warnings
warnings.filterwarnings('ignore')


# TODO: disable validation step


class LitSampleConvNet(pl.LightningModule):
    def __init__(
            self,
            lr: float = 0.1,
            sample_rate: float = 0.001,
            sigma: float = 1.0,
            max_per_sample_grad_norm: float = 1.0,
            delta: float = 1e-5,
            disable_dp: bool = False,
            secure_rng: bool = False,
    ):
        super().__init__()

        # Hyper-parameters
        self.lr = lr
        self.sample_rate = sample_rate
        self.sigma = sigma
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.delta = delta
        self.disable_dp = disable_dp
        self.secure_rng = secure_rng

        if secure_rng:
            try:
                import torchcsprng as prng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e
            self.generator = prng.create_random_device_generator("/dev/urandom")
        else:
            self.generator = None

        # Parameters
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

        # Metrics
        self.test_accuracy = torchmetrics.Accuracy()

        # Set manual optimization mode, otherwise PrivacyEngine crashes
        # TODO: investigate why it doesn't work with automatic_optimization=True
        self.automatic_optimization = False

    def setup(self, stage=None):
        if not self.disable_dp and stage == "fit":
            self.privacy_engine = PrivacyEngine(
                self,
                sample_rate=self.sample_rate,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.sigma,
                max_grad_norm=self.max_per_sample_grad_norm,
                secure_rng=self.secure_rng,
            )

    def teardown(self, stage=None):
        if not self.disable_dp and stage == "fit":
            self.privacy_engine.detach()

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

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0)
        if not self.disable_dp:
            self.privacy_engine.attach(optimizer)
        return optimizer

    def compute_loss(self, batch):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        return loss

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.test_accuracy(output, target)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        return loss


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: Optional[str] = "../mnist",
            sample_rate: float = 0.001,
            test_batch_size: int = 1024,
            secure_rng: bool = False,
    ):
        super().__init__()
        self.data_root = data_dir
        self.dataloader_kwargs = {"num_workers": 1, "pin_memory": True}

        self.save_hyperparameters()

        if secure_rng:
            try:
                import torchcsprng as prng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e
            self.generator = prng.create_random_device_generator("/dev/urandom")
        else:
            self.generator = None

    @property
    def transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self) -> None:
        datasets.MNIST(self.data_root, download=True)

    def train_dataloader(self):
        train_dataset = datasets.MNIST(
            self.data_root,
            train=True,
            download=False,
            transform=self.transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=UniformWithReplacementSampler(
                num_samples=len(train_dataset),
                sample_rate=self.hparams.sample_rate,
                generator=self.generator,
            ),
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST(
                self.data_root,
                train=False,
                download=False,
                transform=self.transform,
            ),
            batch_size=self.hparams.test_batch_size,
            shuffle=True,
            **self.dataloader_kwargs,
        )


def cli_main():
    cli = LightningCLI(
        LitSampleConvNet,
        MNISTDataModule,
        save_config_overwrite=True,
        description="Training MNIST classifier with Opacus and PyTorch Lightning",
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
