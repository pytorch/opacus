#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs MNIST training with differential privacy.
This example demonstrates how to use Opacus with PyTorch Lightning.

To start training:
$ python mnist_lightning.py fit

More information about setting training parameters:
$ python mnist_lightning.py fit --help

To see logs:
$ tensorboard --logdir=lightning_logs/
"""

import warnings
from typing import Optional

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


warnings.filterwarnings("ignore")


class LitSampleConvNetClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.1,
        enable_dp: bool = True,
        delta: float = 1e-5,
        sample_rate: float = 0.001,
        sigma: float = 1.0,
        max_per_sample_grad_norm: float = 1.0,
        secure_rng: bool = False,
    ):
        """A simple conv-net for classifying MNIST with differential privacy

        Args:
            lr: Learning rate
            enable_dp: Enables training with privacy guarantees using Opacus (if True), vanilla SGD otherwise
            delta: Target delta for which (eps, delta)-DP is computed
            sample_rate: Sample rate used for batch construction
            sigma: Noise multiplier
            max_per_sample_grad_norm: Clip per-sample gradients to this norm
            secure_rng: Use secure random number generator
        """
        super().__init__()

        # Hyper-parameters
        self.lr = lr
        self.enable_dp = enable_dp
        self.delta = delta
        self.sample_rate = sample_rate
        self.sigma = sigma
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.secure_rng = secure_rng

        # Parameters
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

        # Privacy engine
        self.privacy_engine = None  # Created before training

        # Metrics
        self.test_accuracy = torchmetrics.Accuracy()

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
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.test_accuracy(output, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        return loss

    # Adding differential privacy learning

    def on_train_start(self) -> None:
        if self.enable_dp:
            self.privacy_engine = PrivacyEngine(
                self,
                sample_rate=self.sample_rate,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.sigma,
                max_grad_norm=self.max_per_sample_grad_norm,
                secure_rng=self.secure_rng,
            )

            optimizer = self.optimizers()
            self.privacy_engine.attach(optimizer)

    def on_train_epoch_end(self):
        if self.enable_dp:
            epsilon, best_alpha = self.privacy_engine.get_privacy_spent(self.delta)
            # Privacy spent: (epsilon, delta) for alpha
            self.log("epsilon", epsilon, on_epoch=True, prog_bar=True)
            self.log("alpha", best_alpha, on_epoch=True, prog_bar=True)

    def on_train_end(self):
        if self.enable_dp:
            self.privacy_engine.detach()


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Optional[str] = "../mnist",
        test_batch_size: int = 1024,
        sample_rate: float = 0.001,
        secure_rng: bool = False,
    ):
        """MNIST DataModule with DP-ready batch sampling

        Args:
            data_dir: A path where MNIST is stored
            test_batch_size: Size of batch for predicting on test
            sample_rate: Sample rate used for batch construction
            secure_rng: Use secure random number generator
        """
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
        return DataLoader(
            train_dataset,
            batch_sampler=UniformWithReplacementSampler(
                num_samples=len(train_dataset),
                sample_rate=self.hparams.sample_rate,
                generator=self.generator,
            ),
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
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
        LitSampleConvNetClassifier,
        MNISTDataModule,
        save_config_overwrite=True,
        trainer_defaults={
            "max_epochs": 10,
            "enable_model_summary": False,
        },
        description="Training MNIST classifier with Opacus and PyTorch Lightning",
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
