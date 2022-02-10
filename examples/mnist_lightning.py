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
Runs MNIST training with differential privacy.
This example demonstrates how to use Opacus with PyTorch Lightning.

To start training:
$ python mnist_lightning.py fit

More information about setting training parameters:
$ python mnist_lightning.py fit --help

To see logs:
$ tensorboard --logdir=lightning_logs/
"""

import os
import warnings

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.lightning import DPLightningDataModule
from pl_bolts.datamodules import MNISTDataModule
from pytorch_lightning.utilities.cli import LightningCLI


warnings.filterwarnings("ignore")


class LitSampleConvNetClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.1,
        enable_dp: bool = True,
        delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
    ):
        """A simple conv-net for classifying MNIST with differential privacy training
        Args:
            lr: Learning rate
            enable_dp: Enables training with privacy guarantees using Opacus (if True), vanilla SGD otherwise
            delta: Target delta for which (eps, delta)-DP is computed
            noise_multiplier: Noise multiplier
            max_grad_norm: Clip per-sample gradients to this norm
        """
        super().__init__()

        # Hyper-parameters
        self.lr = lr

        # Parameters
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

        # Metrics
        self.test_accuracy = torchmetrics.Accuracy()

        # Differential privacy
        self.enable_dp = enable_dp
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        if self.enable_dp:
            self.privacy_engine = PrivacyEngine()

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

        if self.enable_dp:
            data_loader = (
                # soon there will be a fancy way to access train dataloader,
                # see https://github.com/PyTorchLightning/pytorch-lightning/issues/10430
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )

            # transform (model, optimizer, dataloader) to DP-versions
            if hasattr(self, "dp"):
                self.dp["model"].remove_hooks()
            dp_model, optimizer, dataloader = self.privacy_engine.make_private(
                module=self,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=isinstance(data_loader, DPDataLoader),
            )
            self.dp = {"model": dp_model}

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

    def on_train_epoch_end(self):
        # Logging privacy spent: (epsilon, delta)
        epsilon = self.privacy_engine.get_epsilon(self.delta)
        self.log("epsilon", epsilon, on_epoch=True, prog_bar=True)


def main():
    """
    Using vanilla Lightning API to train/test
    """
    data = MNISTDataModule(batch_size=64)
    model = LitSampleConvNetClassifier()

    dp_data = DPLightningDataModule(data)

    trainer = pl.Trainer(
        max_epochs=10,
        enable_model_summary=False,
    )
    trainer.fit(model, dp_data)

    trainer.test(model, data)
    trainer.test(model, dp_data)  # identical


def cli_main():
    """
    Using LightningCLI to automatically setup argparse
    """
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
    if os.environ.get("LIGHTNING_VANILLA") == "true":
        main()
    else:
        cli_main()
