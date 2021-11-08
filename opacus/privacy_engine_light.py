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
        sample_rate: float = 0.001,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.sample_rate = sample_rate
        self.generator = generator

    def wrap_dataloader(self, dataloader: DataLoader) -> DataLoader:
        batch_sampler = UniformWithReplacementSampler(
            num_samples=len(dataloader.dataset),
            sample_rate=self.sample_rate,
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

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule.setup(stage)

    def train_dataloader(self):
        dataloader = self.datamodule.train_dataloader()
        return self.wrap_dataloader(dataloader)

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
        privacy_engine: PrivacyEngine,

    ):
        """Callback enabling differential privacy learning in PyTorch Lightning trainer

        Args:
            privacy_engine:
        """
        self.privacy_engine = privacy_engine

        self.original_dataloader = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        optimizer = pl_module.optimizers()

        model, optimizer, data_loader = self.privacy_engine.make_private(pl_module, optimizer, trainer.train_dataloader.lia)

        pl_module.privacy_engine = PrivacyEngine(
            pl_module,
            sample_rate=self.sample_rate,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=self.sigma,
            max_grad_norm=self.max_per_sample_grad_norm,
            secure_rng=self.secure_rng,
        )


        pl_module.privacy_engine.attach(optimizer.optimizer)

        # TODO: check if data loader is DP-compatible

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epsilon, best_alpha = pl_module.privacy_engine.get_privacy_spent(self.delta)
        # Privacy spent: (epsilon, delta) for alpha
        pl_module.log("epsilon", epsilon, on_epoch=True, prog_bar=True)
        pl_module.log("alpha", best_alpha, on_epoch=True, prog_bar=True)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.privacy_engine.detach()

    def wrap_datamodule(self, datamodule: pl.LightningDataModule) -> DPLightningDataModule:
        return DPLightningDataModule(
            datamodule,
            sample_rate=self.sample_rate,
            generator=self.generator,
        )


class PrivacyEngine:
    def __init__(
        self,
        accountant: Union[Type[IAccountant], str] = RDPAccountant,
        poisson_sampling: bool = True,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        try_fix_incompatible_modules: bool = False,
    ):
        accountant_cls = ACCOUNTANTS[accountant] if isinstance(accountant, str) else accountant
        self.accountant = accountant_cls()
        self.poisson_sampling = poisson_sampling
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.try_fix_incompatible_modules = try_fix_incompatible_modules

    def make_private(
        self,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
    ):
        distributed = type(module) is DPDDP

        # (fix and) validate
        if self.try_fix_incompatible_modules:
            module = ModuleValidator.fix(module)
        ModuleValidator.validate(module, raise_if_error=True)

        # wrap module
        if not isinstance(module, GradSampleModule):
            module = GradSampleModule(
                module,
                batch_first=self.batch_first,
                loss_reduction=self.loss_reduction,
            )

        # TODO: either validate consistent dataset or do per-dataset accounting
        if self.poisson_sampling:
            data_loader = DPDataLoader.from_data_loader(data_loader, distributed=distributed)

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        if isinstance(optimizer, DPOptimizer):
            # TODO: lol rename optimizer optimizer optimizer
            optimizer = optimizer.optimizer

        optimizer_cls = DPOptimizer if not distributed else DistributedDPOptimizer
        optimizer = optimizer_cls(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=self.loss_reduction,
        )

        def accountant_hook(optim: DPOptimizer):
            # TODO: This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in
            # distributed mode, each node samples among a subset of the data)
            self.accountant.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        optimizer.attach_step_hook(accountant_hook)

        return module, optimizer, data_loader


    def wrap_datamodule(self, datamodule):
        raise NotImplementedError

    def lightning_callback(self):
        raise NotImplementedError

