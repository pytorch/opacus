#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import types
import warnings
from typing import List, Union

import torch
from torch import nn

from . import privacy_analysis as tf_privacy, utils
from .dp_model_inspector import DPModelInspector
from .per_sample_gradient_clip import PerSampleGradientClipper


class PrivacyEngine:
    def __init__(
        self,
        module: nn.Module,
        batch_size: int,
        sample_size: int,
        alphas: List[float],
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        grad_norm_type: int = 2,
        batch_dim: int = 0,
        target_delta: float = 1e-6,
        loss_reduction: str = "mean",
        **misc_settings,
    ):
        self.steps = 0
        self.module = module
        self.alphas = alphas
        self.device = next(module.parameters()).device

        self.batch_size = batch_size
        self.sample_rate = batch_size / sample_size
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.grad_norm_type = grad_norm_type
        self.batch_dim = batch_dim
        self.target_delta = target_delta

        self._set_seed(None)
        self.validator = DPModelInspector()
        self.clipper = None  # lazy initialization in attach
        self.misc_settings = misc_settings

        self.loss_reduction = loss_reduction

    def detach(self):
        optim = self.optimizer
        optim.privacy_engine = None
        self.clipper.close()
        optim.step = types.MethodType(optim.original_step, optim)
        del optim.virtual_step

    def attach(self, optimizer: torch.optim.Optimizer):
        """
        Attaches to a `torch.optim.Optimizer` object, and injects itself into
        the optimizer's step.

        To do that, this method does the following:
        1. Validates the model for containing un-attachable layers
        2. Adds a pointer to this object (the PrivacyEngine) inside the optimizer
        3. Moves the original optimizer's `step()` function to `original_step()`
        4. Monkeypatches the optimizer's `step()` function to call `step()` on
           the query engine automatically whenever it would call `step()` for itself
        """

        # Validate the model for not containing un-supported modules.
        self.validator.validate(self.module)
        # only attach if model is validated
        norm_clipper = (
            utils.ConstantFlatClipper(self.max_grad_norm)
            if not isinstance(self.max_grad_norm, list)
            else utils.ConstantPerLayerClipper(self.max_grad_norm)
        )
        if self.misc_settings.get('experimental', False):
            norm_clipper = utils._Experimental_Clipper_(
                [self.max_grad_norm],
                self.misc_settings.get('clip_per_layer', False),
                self.misc_settings.get('clipping_method', utils.ClippingMethod.STATIC),
                self.misc_settings.get('ratio', 0.0)
            )

        self.clipper = PerSampleGradientClipper(
            self.module,
            norm_clipper,
            self.batch_dim == 0,
        )

        def dp_step(self, closure=None):
            self.privacy_engine.step()
            self.original_step(closure)

        optimizer.privacy_engine = self
        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)

        # We add a 'virtual_step' function to the optimizer, which
        # enables the use of virtual batches.
        # By repeatedly computing backward passes and calling virtual_step,
        # we can aggregate the clipped gradient for large batches
        def virtual_step(self):
            self.privacy_engine.virtual_step()

        optimizer.virtual_step = types.MethodType(virtual_step, optimizer)

        self.optimizer = optimizer  # create a cross reference for detaching

    def get_renyi_divergence(self):
        rdp = torch.tensor(
            tf_privacy.compute_rdp(
                self.sample_rate, self.noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    def get_privacy_spent(self, target_delta: float = None):
        if target_delta is None:
            target_delta = self.target_delta
        rdp = self.get_renyi_divergence() * self.steps
        return tf_privacy.get_privacy_spent(self.alphas, rdp, target_delta)

    def step(self):
        self.steps += 1
        self.clipper.clip_and_accumulate()
        clip_values, batch_size = self.clipper.pre_step()

        # ensure the clipper consumed the right amount of gradients.
        # In the last batch of a training epoch, we might get a batch that is
        # smaller than others but we should never get a batch that is too large
        if batch_size > self.batch_size:
            raise ValueError(
                f"PrivacyEngine expected a batch of size {self.batch_size} "
                f"but received a batch of size {batch_size}"
            )

        if batch_size < self.batch_size:
            warnings.warn(
                f"PrivacyEngine expected a batch of size {self.batch_size} "
                f"but the last step received a batch of size {batch_size}. "
                "This means that the privacy analysis will be a bit more "
                "pessimistic. You can set `drop_last = True` in your PyTorch "
                "dataloader to avoid this problem completely"
            )

        params = (p for p in self.module.parameters() if p.requires_grad)
        for p, clip_value in zip(params, clip_values):
            noise = self._generate_noise(clip_value, p)
            if self.loss_reduction == "mean":
                noise /= batch_size
            p.grad += noise

    def to(self, device):
        self.device = device
        return self

    def virtual_step(self):
        self.clipper.clip_and_accumulate()

    def _generate_noise(self, max_norm, parameter):
        if self.noise_multiplier > 0:
            return torch.normal(
                0,
                self.noise_multiplier * max_norm,
                parameter.grad.shape,
                device=self.device,
                generator=self.secure_generator,
            )
        return 0.0

    def _set_seed(self, secure_seed: int):
        if secure_seed is not None:
            self.secure_seed = secure_seed
        else:
            self.secure_seed = int.from_bytes(
                os.urandom(8), byteorder="big", signed=True
            )
        self.secure_generator = (
            torch.random.manual_seed(self.secure_seed)
            if self.device.type == "cpu"
            else torch.cuda.manual_seed(self.secure_seed)
        )
