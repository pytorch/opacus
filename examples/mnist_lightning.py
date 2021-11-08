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
from typing import Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.grad_sample import GradSampleModule
from opacus.lightning import DPLightningDataModule
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pl_bolts.datamodules import MNISTDataModule

from opacus.optimizers import DPOptimizer

warnings.filterwarnings("ignore")


class LitSampleConvNetClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.1,
    ):
        """A simple conv-net for classifying MNIST
        Args:
            lr: Learning rate
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

        self.delta = 1e-5
        self.accountant = RDPAccountant()

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

    def on_train_start(self) -> None:
        print("on_train_start")

    def setup(self, stage):
        if stage == 'fit':
            print("setup", self.trainer._data_connector._train_dataloader_source.is_defined())


    def configure_optimizers(self):
        print("configure_optimizers")

        print("train_dataloader:", self.trainer._data_connector._train_dataloader_source.is_defined())
        data_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        dp_model = GradSampleModule(self)
        self.dp_model = {"dp_model": dp_model}
        optimizer = optim.SGD(dp_model.parameters(), lr=self.lr, momentum=0)
        dp_optimizer = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            expected_batch_size=expected_batch_size,  # requires data
        )

        sample_rate = 0.01  # requires data

        def accountant_hook(optim: DPOptimizer):
            self.accountant.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        dp_optimizer.attach_step_hook(accountant_hook)

        return dp_optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.dp_model["dp_model"](data)
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
        epsilon, best_alpha = self.accountant.get_privacy_spent(self.delta)
        # Privacy spent: (epsilon, delta) for alpha
        self.log("epsilon", epsilon, on_epoch=True, prog_bar=True)
        self.log("alpha", best_alpha, on_epoch=True, prog_bar=True)


def main():
    # Look ma, no privacy burden here!
    data = MNISTDataModule()
    model = LitSampleConvNetClassifier()

    # Here we add some privacy
    # accountant = RDPAccountant()
    # privacy_engine_callback = PrivacyEngineCallback(
    #     accountant=accountant,
    #     delta=1e-5,
    #     noise_multiplier=1.0,
    #     max_grad_norm=1.0,
    # )

    dp_data = DPLightningDataModule(data)

    # Now we go
    trainer = pl.Trainer(
        max_epochs=2,
        enable_model_summary=False,
    )
    trainer.fit(model, dp_data)

    # TODO:
    # trainer.fit(model, data)
    # Must crash with the message: either remove PrivacyEngine or use certified data modules

    trainer.test(model, data)
    trainer.test(model, dp_data)  # identical


def cli_main():
    # TODO: add optional LightningPrivacyEngine() callback
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
    #cli_main()
    main()