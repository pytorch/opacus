#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
from opacus import PerSampleGradientClipper
from opacus.layers import SequenceBias
from opacus.utils.clipping import ConstantFlatClipper


class PerSampleGradientTest(unittest.TestCase):
    """
    Checks the correctness of per sample gradient computation by splitting the data
    into batches of size 1 and computing vanilla gradients
    """

    def _run_once(self, layer, criterion, data):
        layer.zero_grad()
        output = layer(data).squeeze()

        y = torch.zeros_like(output)
        loss = criterion(output, y)
        loss.backward()

    def _check_one_layer(self, layer, data, batch_first=True):
        self._check_one_layer_with_criterion(
            layer, nn.L1Loss(reduction="mean"), data, batch_first
        )
        self._check_one_layer_with_criterion(
            layer, nn.L1Loss(reduction="sum"), data, batch_first
        )

    def _check_one_layer_with_criterion(self, layer, criterion, data, batch_first=True):
        clipper = PerSampleGradientClipper(
            layer,
            ConstantFlatClipper(1e9),
            batch_first=batch_first,
            loss_reduction=criterion.reduction,
        )
        self._run_once(layer, criterion, data)

        computed_sample_grads = {}
        for (param_name, param) in layer.named_parameters():
            computed_sample_grads[param_name] = param.grad_sample.detach()

        clipper.clip_and_accumulate()
        clipper.pre_step()
        clipper.close()

        batch_dim = 0 if batch_first else 1
        data = data.transpose(0, batch_dim)
        for i, sample in enumerate(data):
            # simulate batch_size = 1
            sample_data = sample.unsqueeze(batch_dim)
            self._run_once(layer, criterion, sample_data)

            for (param_name, param) in layer.named_parameters():
                # grad we just computed with batch_size = 1
                vanilla_per_sample_grad = param.grad

                # i-th line in grad_sample computed before
                computed_per_sample_grad = computed_sample_grads[param_name][i]

                self.assertTrue(
                    torch.allclose(
                        vanilla_per_sample_grad,
                        computed_per_sample_grad,
                        atol=10e-5,
                        rtol=10e-3,
                    ),
                    f"Gradient mismatch. Parameter: {layer}.{param_name}, loss: {criterion.reduction}",
                )

    def test_conv1d(self):
        x = torch.randn(24, 16, 24)
        layer = nn.Conv1d(16, 32, 3, 1)

        self._check_one_layer(layer, x)

    def test_conv2d(self):
        x = torch.randn(24, 16, 24, 24)
        layer = nn.Conv2d(16, 32, 3, 1)

        self._check_one_layer(layer, x)

    def test_linear(self):
        self._check_one_layer(nn.Linear(8, 4), torch.randn(16, 8))
        self._check_one_layer(nn.Linear(8, 4), torch.randn(24, 8, 8))

    def test_layernorm(self):
        x = torch.randn(64, 16, 24, 24)

        self._check_one_layer(nn.LayerNorm(24), x)
        self._check_one_layer(nn.LayerNorm((24, 24)), x)
        self._check_one_layer(nn.LayerNorm((16, 24, 24)), x)

    def test_groupnorm(self):
        self._check_one_layer(nn.GroupNorm(4, 16), torch.randn(16, 16, 10))
        self._check_one_layer(nn.GroupNorm(4, 16), torch.randn(16, 16, 10, 9))
        self._check_one_layer(nn.GroupNorm(4, 16), torch.randn(16, 16, 10, 9, 8))

    def test_instancenorm(self):
        self._check_one_layer(
            nn.InstanceNorm1d(16, affine=True), torch.randn(16, 16, 10)
        )
        self._check_one_layer(
            nn.InstanceNorm2d(16, affine=True), torch.randn(16, 16, 10, 9)
        )
        self._check_one_layer(
            nn.InstanceNorm3d(16, affine=True), torch.randn(16, 16, 10, 9, 8)
        )

    def test_sequence_bias(self):
        x = torch.randn(4, 3, 2)
        layer = SequenceBias(2)

        self._check_one_layer(layer, x, batch_first=False)

    def test_embedding(self):
        layer = nn.Embedding(256, 100)
        x1 = torch.randint(0, 255, (24, 42)).long()
        x2 = torch.randint(0, 255, (12, 64, 4)).long()
        self._check_one_layer(layer, x1)
        self._check_one_layer(layer, x2)
