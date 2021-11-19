#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import warnings
from typing import List, Optional, Union

import torch
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader, switch_generator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers import DPOptimizer, get_optimizer_class
from opacus.validators.module_validator import ModuleValidator
from torch import nn, optim
from torch.utils.data import DataLoader


def forbid_accumulation_hook(module: nn.Module, _):
    if not module.training:
        return

    for p in module.parameters():
        if hasattr(p, "grad_sample"):
            raise ValueError(
                "Poisson sampling is not compatible with grad accumulation. "
                "You need to call optimizer.step() after every forward/backward pass "
                "or consider using BatchMemoryManager"
            )


class PrivacyEngine:
    def __init__(self, accountant: str = "rdp", secure_mode=False):
        """
        # TODO: Add docstring with doctest
        # - Creating PrivacyEngine and applying make_private (test_privacy_engine_class_example)
        # - Moving model to another device
        # - Virtual step

        Example:
            >>> dataloader = getfixture("demo_dataloader")  # doctest: +SKIP
            >>> criterion = nn.CrossEntropyLoss()  # doctest: +SKIP
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # doctest: +SKIP
            >>> privacy_engine = PrivacyEngine()  # doctest: +SKIP
            >>> for i, (X, y) in enumerate(dataloader):  # doctest: +SKIP
            ...     logits = model(X)
            ...     loss = criterion(logits, y)
            ...     loss.backward()
            ...     if i % 16 == 15:
            ...         optimizer.step()  # this will call privacy engine's step()
            ...         optimizer.zero_grad()
            ...     else:
            ...         optimizer.virtual_step()  # this will call privacy engine's virtual_step()

        """
        self.accountant = create_accountant(mechanism=accountant)
        self.secure_mode = secure_mode
        self.secure_rng = None
        self.dataset = None  # only used to detect switching to a different dataset

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

    def _prepare_optimizer(
        self,
        *,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        noise_seed: Optional[int] = None,
        distributed: bool = False,
        clipping: str = "flat",
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(noise_seed)

        optim_class = get_optimizer_class(clipping=clipping, distributed=distributed)

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
        )

    def _prepare_data_loader(
        self,
        data_loader: DataLoader,
        poisson_sampling: bool,
        distributed: bool,
    ) -> DataLoader:
        if self.dataset is None:
            self.dataset = data_loader.dataset
        elif self.dataset != data_loader.dataset:
            warnings.warn(
                f"PrivacyEngine detected new dataset object. "
                f"Was: {self.dataset}, got: {data_loader.dataset}. "
                f"Privacy accounting works per dataset, please initialize "
                f"new PrivacyEngine if you're using different dataset. "
                f"You can ignore this warning if two datasets above "
                f"represent the same logical dataset"
            )

        if poisson_sampling:
            return DPDataLoader.from_data_loader(
                data_loader, generator=self.secure_rng, distributed=distributed
            )
        elif self.secure_mode:
            return switch_generator(data_loader, self.secure_rng)
        else:
            return data_loader

    def _prepare_model(
        self, module: nn.Module, batch_first: bool = True, loss_reduction: str = "mean"
    ) -> GradSampleModule:
        # Ideally, validation should have been taken care of by calling
        # `get_compatible_module()`
        self.validate(module=module, optimizer=None, data_loader=None)

        # wrap
        if isinstance(module, GradSampleModule):
            if (
                module.batch_first != batch_first
                or module.loss_reduction != loss_reduction
            ):
                raise ValueError(
                    f"Pre-existing GradSampleModule doesn't match new arguments."
                    f"Got: module.batch_first: {module.batch_first}, module.loss_reduction: {module.loss_reduction}"
                    f"Requested: batch_first:{batch_first}, loss_reduction: {loss_reduction}. "
                    f"Please pass vanilla nn.Module instead"
                )

            return module
        else:
            return GradSampleModule(
                module, batch_first=batch_first, loss_reduction=loss_reduction
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

    @classmethod
    def get_compatible_module(cls, module: nn.Module) -> nn.Module:
        """
        Return a privacy engine compatible module. Also validates the module after
        running registered fixes.
        """
        module = ModuleValidator.fix(module)
        ModuleValidator.validate(module, raise_if_error=True)
        return module

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        noise_seed: Optional[int] = None,
        poisson_sampling: bool = True,
        clipping: str = "flat",
    ):
        if noise_seed and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        distributed = type(module) is DPDDP

        module = self._prepare_model(module, batch_first, loss_reduction)
        if poisson_sampling:
            module.register_forward_pre_hook(forbid_accumulation_hook)

        data_loader = self._prepare_data_loader(
            data_loader, distributed=distributed, poisson_sampling=poisson_sampling
        )

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
            clipping=clipping,
        )

        def accountant_hook(optim: DPOptimizer):
            # This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in
            # distributed mode, each node samples among a subset of the data)
            self.accountant.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        optimizer.attach_step_hook(accountant_hook)

        return module, optimizer, data_loader

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
        **kwargs,
    ):
        sample_rate = 1 / len(data_loader)

        if len(self.accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_epsilon with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                accountant=self.accountant.mechanism(),
                **kwargs,
            ),
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_seed=noise_seed,
        )

    def get_epsilon(self, delta):
        return self.accountant.get_epsilon(delta)