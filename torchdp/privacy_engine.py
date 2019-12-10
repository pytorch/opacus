#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import types
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import privacy_analysis as tf_privacy
from .per_sample_gradient_clip import PerSampleGradientClipper


class PrivacyEngine:
    def __init__(
        self,
        module: nn.Module,
        dataloader: DataLoader,
        alphas: List[float],
        noise_multiplier: float,
        max_grad_norm: float,
        grad_norm_type: int = 2,
    ):
        self.steps = 0
        self.module = module
        self.dataloader = dataloader
        self.alphas = alphas
        self.device = next(module.parameters()).device

        self.sample_rate = dataloader.batch_size / len(dataloader.dataset)
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.grad_norm_type = grad_norm_type

        self.secure_seed = int.from_bytes(os.urandom(8), byteorder="big", signed=True)
        self.secure_generator = (
            torch.random.manual_seed(self.secure_seed)
            if self.device.type == "cpu"
            else torch.cuda.manual_seed(self.secure_seed)
        )

        self.clipper = PerSampleGradientClipper(self.module, self.max_grad_norm)

    def attach(self, optimizer: torch.optim.Optimizer):
        """
        Attaches to a `torch.optim.Optimizer` object, and injects itself into
        the optimizer's step.

        To do that, this method does the following:
        1. Adds a pointer to this object (the PrivacyEngine) inside the optimizer
        2. Moves the original optimizer's `step()` function to `original_step()`
        3. Monkeypatches the optimizer's `step()` function to call `step()` on
        the query engine automatically whenever it would call `step()` for itself
        """

        def dp_step(self, closure=None):
            self.privacy_engine.step()
            self.original_step(closure)

        optimizer.privacy_engine = self
        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)

    def get_renyi_divergence(self):
        rdp = torch.tensor(
            tf_privacy.compute_rdp(
                self.sample_rate, self.noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    def get_privacy_spent(self, target_delta: float):
        rdp = self.get_renyi_divergence() * self.steps
        return tf_privacy.get_privacy_spent(
            self.alphas, rdp, target_delta
        )

    def step(self):
        self.steps += 1
        self.clipper.step()

        for p in self.module.parameters():
            noise = torch.normal(
                0,
                self.noise_multiplier * self.max_grad_norm,
                p.grad.shape,
                device=self.device,
                generator=self.secure_generator,
            )
            p.grad += noise / self.clipper.batch_size

    def to(self, device):
        self.device = device
        return self
