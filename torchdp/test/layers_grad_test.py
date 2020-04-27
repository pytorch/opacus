#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
import torch.nn as nn
from torchdp import PerSampleGradientClipper
from torchdp.dp_model_inspector import DPModelInspector
from torchdp.layers import DPMultiheadAttention, SequenceBias


class LayersGradTest(unittest.TestCase):
    def setUp(self):
        self.validator = DPModelInspector()

    def _reset_seeds(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)

    def _check_one_layer(self, layer, *args, **kwargs):
        if hasattr(layer, "autograd_grad_sample_hooks"):
            raise ValueError(
                f"Input layer already has hooks attached."
                f"Please provide freshly constructed layer"
            )

        self.validator.validate(layer)
        if hasattr(layer, "weight"):
            nn.init.uniform_(layer.weight)
        if hasattr(layer, "bias"):
            nn.init.uniform_(layer.bias)

        self._reset_seeds()
        output = layer(*args)
        if isinstance(output, tuple):
            output = output[0]
        output.norm().backward()
        vanilla_run_grads = [
            p.grad.detach().clone() for p in layer.parameters() if p.requires_grad
        ]

        clipper = PerSampleGradientClipper(
            layer, 999, batch_dim=kwargs.get("batch_dim", 0)
        )
        self._reset_seeds()
        output = layer(*args)
        if isinstance(output, tuple):
            output = output[0]
        output.norm().backward()
        clipper.step()

        for param_name, param in layer.named_parameters():
            if param.requires_grad:
                self.assertTrue(
                    hasattr(param, "grad_sample"),
                    f"Per-sample gradients hasn't been computed for {param_name}",
                )

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

    def test_sequence_bias(self):
        x = torch.randn(4, 3, 2)
        layer = SequenceBias(2)

        self._check_one_layer(layer, x, batch_dim=1)

    def test_multihead_attention(self):
        x = torch.randn(16, 24, 32)

        layer = DPMultiheadAttention(32, 1)
        self._check_one_layer(layer, x, x, x, batch_dim=1)

        layer = DPMultiheadAttention(32, 1, bias=True, add_bias_kv=True, dropout=0.05)
        self._check_one_layer(layer, x, x, x, batch_dim=1)

        layer = DPMultiheadAttention(32, 1, bias=True, add_bias_kv=True)
        self._check_one_layer(layer, x, x, x, batch_dim=1)

        layer = DPMultiheadAttention(
            32, 1, bias=True, add_bias_kv=True, add_zero_attn=True
        )
        self._check_one_layer(layer, x, x, x, batch_dim=1)

        q = torch.randn(16, 24, 32)
        k = torch.randn(20, 24, 28)
        v = torch.randn(20, 24, 28)
        layer = DPMultiheadAttention(
            32, 1, bias=True, add_bias_kv=True, add_zero_attn=True, kdim=28, vdim=28
        )
        self._check_one_layer(layer, q, k, v, batch_dim=1)
