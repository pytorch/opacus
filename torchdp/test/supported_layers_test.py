#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
from torchdp import PerSampleGradientClipper


class SupportedLayersTest(unittest.TestCase):

    def _check_one_layer(self, layer, input):
        if hasattr(layer, "autograd_grad_sample_hooks"):
            raise ValueError(
                f"Input layer already has hooks attached."
                f"Please provide freshly constructed layer"
            )

        nn.init.uniform_(layer.weight)
        nn.init.uniform_(layer.bias)

        output = layer(input)
        output.norm().backward()
        vanilla_run_grads = [
            p.grad.detach().clone() for p in layer.parameters() if p.requires_grad
        ]

        clipper = PerSampleGradientClipper(layer, 999)
        output = layer(input)
        output.norm().backward()
        clipper.step()
        private_run_grads = [
            p.grad.detach().clone() for p in layer.parameters() if p.requires_grad
        ]

        for vanilla_grad, private_grad in zip(vanilla_run_grads, private_run_grads):
            self.assertTrue(
                torch.allclose(vanilla_grad, private_grad, atol=10e-5, rtol=10e-3)
            )

    def test_conv1d(self):
        x = torch.randn(64, 16, 24)
        layer = nn.Conv1d(16, 32, 3, 1)

        self._check_one_layer(layer, x)

    def test_conv2d(self):
        x = torch.randn(64, 16, 24, 24)
        layer = nn.Conv2d(16, 32, 3, 1)

        self._check_one_layer(layer, x)

    def test_linear(self):
        self._check_one_layer(nn.Linear(8, 4), torch.randn(16, 8))
        self._check_one_layer(nn.Linear(8, 4), torch.randn(16, 8, 8))

    def test_layernorm(self):
        x = torch.randn(64, 16, 24, 24)

        self._check_one_layer(nn.LayerNorm(24), x)
        self._check_one_layer(nn.LayerNorm((24, 24)), x)
        self._check_one_layer(nn.LayerNorm((16, 24, 24)), x)

    def test_groupnorm(self):
        self._check_one_layer(nn.GroupNorm(4, 16), torch.randn(64, 16, 10))
        self._check_one_layer(nn.GroupNorm(4, 16), torch.randn(64, 16, 10, 9))
        self._check_one_layer(nn.GroupNorm(4, 16), torch.randn(64, 16, 10, 9, 8))

    def test_instancenorm(self):
        self._check_one_layer(
            nn.InstanceNorm1d(16, affine=True), torch.randn(64, 16, 10)
        )
        self._check_one_layer(
            nn.InstanceNorm2d(16, affine=True), torch.randn(64, 16, 10, 9)
        )
        self._check_one_layer(
            nn.InstanceNorm3d(16, affine=True), torch.randn(64, 16, 10, 9, 8)
        )
