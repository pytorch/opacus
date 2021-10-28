#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import warnings
from typing import List, Optional

import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.rdp import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers import (
    DistributedDPOptimizer,
    DistributedPerLayerOptimizer,
    DPOptimizer,
    DPPerLayerOptimizer,
)
from opacus.validators.module_validator import ModuleValidator
from torch import nn, optim
from torch.utils.data import DataLoader


def forbid_accumulation_hook(module: nn.Module, _):
    for p in module.parameters():
        if hasattr(p, "grad_sample"):
            # TODO: this correspond to either not calling optimizer.step()
            # or not calling zero_grad(). Customize message
            raise ValueError(
                "Poisson sampling is not compatible with grad accumulation"
            )


class PrivacyEngine:
    def __init__(self, secure_mode=False):
        self.accountant = RDPAccountant()
        self.secure_mode = secure_mode

        if self.secure_mode:
            try:
                import torchcsprng as csprng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            self.secure_rng = csprng.create_random_device_generator("/dev/urandom")
        else:
            warnings.warn(
                "Secure RNG turned off. This is perfectly fine for experimentation as it allows "
                "for much faster training performance, but remember to turn it on and retrain "
                "one last time before production with ``secure_mode`` turned on."
            )

    def is_compatible(
        self,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ) -> bool:
        """
        Check if task components are compatible with DP.
        """
        return ModuleValidator.is_valid(module)

    def validate(
        self,
        module: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: Optional[DataLoader],
    ):
        """
        Validate that task components are compatible with DP.
        Same as ``is_compatible()``, but raises error instead of returning bool.
        """
        ModuleValidator.validate(module, raise_if_error=True)

    # TODO: add * syntax for keyword args
    def make_private(
        self,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: float,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        noise_seed: Optional[int] = None,
        poisson_sampling: bool = True,
        try_fix_incompatible_modules: bool = False,
    ):
        distributed = type(module) is DPDDP

        if noise_seed and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        module = self._prepare_model(
            module, batch_first, loss_reduction, try_fix_incompatible_modules
        )
        # TODO: either validate consistent dataset or do per-dataset accounting
        if poisson_sampling:
            data_loader = self._prepare_data_loader(
                data_loader, distributed=distributed
            )
            module.register_forward_pre_hook(forbid_accumulation_hook)

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_seed=noise_seed,
            distributed=distributed,
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

    def make_private_per_layer(
        self,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norms: float,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        noise_seed: Optional[int] = None,
        poisson_sampling: bool = True,
        try_fix_incompatible_modules: bool = False,
    ):
        distributed = type(module) is DPDDP

        if noise_seed and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        module = self._prepare_model(
            module, batch_first, loss_reduction, try_fix_incompatible_modules
        )

        # TODO: either validate consistent dataset or do per-dataset accounting
        module = self._prepare_model(
            module,
            batch_first,
            loss_reduction,
        )
        if poisson_sampling:
            data_loader = self._prepare_data_loader(
                data_loader, distributed=distributed
            )
            module.register_forward_pre_hook(forbid_accumulation_hook)

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        optimizer = self._prepare_optimizer_per_layer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norms=max_grad_norms,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_seed=noise_seed,
            distributed=distributed,
        )

        if distributed:
            optimizer.register_hooks(module)

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

    # TODO: we need a test for that
    def make_private_with_epsilon(
        self,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: float,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        noise_seed: Optional[int] = None,
        try_fix_incompatible_modules: bool = False,
        alphas: Optional[List[float]] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
    ):
        sample_rate = 1 / len(data_loader)

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                alphas=alphas,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            ),
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_seed=noise_seed,
            try_fix_incompatible_modules=try_fix_incompatible_modules,
        )

    def _prepare_model(
        self,
        module: nn.Module,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        try_fix_incompatible_modules: bool = False,
    ) -> GradSampleModule:
        # (fix and) validate
        if try_fix_incompatible_modules:
            module = ModuleValidator.fix(module)
        self.validate(module=module, optimizer=None, data_loader=None)

        # wrap
        if isinstance(module, GradSampleModule):
            return module
        else:
            return GradSampleModule(
                module, batch_first=batch_first, loss_reduction=loss_reduction
            )

    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        loss_reduction: str = "mean",
        noise_seed: Optional[int] = None,
        distributed: bool = False,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            # TODO: lol rename optimizer optimizer optimizer
            optimizer = optimizer.optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(noise_seed)

        if distributed:
            return DistributedDPOptimizer(
                optimizer=optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                loss_reduction=loss_reduction,
                generator=generator,
            )
        else:
            return DPOptimizer(
                optimizer=optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                loss_reduction=loss_reduction,
                generator=generator,
            )

    def _prepare_optimizer_per_layer(
        self,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norms: float,
        expected_batch_size: int,
        loss_reduction: str = "mean",
        noise_seed: Optional[int] = None,
        distributed: bool = False,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            # TODO: lol rename optimizer optimizer optimizer
            optimizer = optimizer.optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(noise_seed)

        optim_class = (
            DistributedPerLayerOptimizer if distributed else DPPerLayerOptimizer
        )

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norms=max_grad_norms,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
        )

    def _prepare_data_loader(
        self, data_loader: DataLoader, distributed: bool
    ) -> DataLoader:
        if isinstance(data_loader, DPDataLoader):
            return data_loader

        generator = None
        if self.secure_mode:
            generator = self.secure_rng

        return DPDataLoader.from_data_loader(
            data_loader, generator=generator, distributed=distributed
        )

    # TODO: default delta value?
    def get_epsilon(self, delta, alphas=None):
        return self.accountant.get_privacy_spent(delta)[0]
