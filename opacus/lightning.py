#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional, Type, Union

from opacus.accountants import RDPAccountant, IAccountant, GaussianAccountant
from opacus.accountants.rdp import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers import (
    DPOptimizer,
    DistributedDPOptimizer,
    DPPerLayerOptimizer,
    DistributedPerLayerOptimizer,
)
from opacus.validators.module_validator import ModuleValidator
from torch import nn, optim
from torch.utils.data import DataLoader

from typing import Optional, Any

import pytorch_lightning as pl
import torch
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


ACCOUNTANTS = {
    "rdp": RDPAccountant,
    "gaussian": GaussianAccountant,
}



class DPLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datamodule: pl.LightningDataModule,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.generator = generator

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule.setup(stage)

    def train_dataloader(self):
        dataloader = self.datamodule.train_dataloader()

        batch_sampler = UniformWithReplacementSampler(
            num_samples=len(dataloader.dataset),
            sample_rate=1 / len(dataloader),
            generator=self.generator,
        )

        return DataLoader(
            # changed by the wrapper
            generator=self.generator,
            batch_sampler=batch_sampler,
            # inherited from the object
            dataset=dataloader.dataset,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn, # wrap_collate_with_empty(collate_fn, sample_empty_shapes),
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            multiprocessing_context=dataloader.multiprocessing_context,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
        )

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def predict_dataloader(self):
        return self.datamodule.predict_dataloader()

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        return self.datamodule.transfer_batch_to_device(batch, device, dataloader_idx)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.datamodule.on_before_batch_transfer(batch, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.datamodule.on_after_batch_transfer(batch, dataloader_idx)


class PrivacyEngineCallback(pl.Callback):

    def __init__(
        self,
        accountant: RDPAccountant,
        delta: float,

        noise_multiplier: float,
        max_grad_norm: float,
        loss_reduction: str = "mean",
    ):
        """Callback enabling differential privacy learning in PyTorch Lightning trainer
        """
        self.accountant = accountant
        self.delta = delta

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.loss_reduction = loss_reduction

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # wrap module
        #dp_module = GradSampleModule(pl_module)
        #trainer.model = dp_module

        # inspect train dataloader
        assert trainer._data_connector._train_dataloader_source.is_defined()
        dataloader = trainer._data_connector._train_dataloader_source.dataloader()
        # TODO: check poisson sampling

        sample_rate = 1 / len(dataloader)
        expected_batch_size = int(len(dataloader.dataset) * sample_rate)

        # wrap optimizer
        original_optimizer = trainer.optimizers
        optimizer = DPOptimizer(
            optimizer=original_optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            loss_reduction=self.loss_reduction,
            expected_batch_size=expected_batch_size,
        )
        trainer.optimizers = optimizer

        def accountant_hook(optim: DPOptimizer):
            # TODO: This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in
            # distributed mode, each node samples among a subset of the data)
            self.accountant.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        optimizer.attach_step_hook(accountant_hook)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epsilon, best_alpha = self.accountant.get_privacy_spent(self.delta)[0]
        # Privacy spent: (epsilon, delta) for alpha
        pl_module.log("epsilon", epsilon, on_epoch=True, prog_bar=True)
        pl_module.log("alpha", best_alpha, on_epoch=True, prog_bar=True)

