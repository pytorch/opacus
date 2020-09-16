#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import unittest

import torch
import torch.nn as nn
from opacus.dp_module.module_checkers.errors import ShouldReplaceModuleError
from opacus.dp_module.module_checkers.multihead_attention import (
    DPMultiheadAttention,
    MultiheadAttentionChecker,
)


class DPMultiheadAttention_test(unittest.TestCase):
    def setUp(self):
        self.EMBED_SIZE = 32

    def _reset_seeds(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)

    def _run_multihead(self, q, k, v, **kwargs):
        original_layer = nn.MultiheadAttention(self.EMBED_SIZE, **kwargs)
        dp_layer = DPMultiheadAttention(self.EMBED_SIZE, **kwargs)
        dp_layer.load_state_dict(original_layer.state_dict())

        self._reset_seeds()
        original_y, original_attn_weights = original_layer(q, k, v)

        self._reset_seeds()
        dp_y, dp_attn_weights = dp_layer(q, k, v)

        self.assertTrue(torch.allclose(original_y, dp_y, atol=10e-4, rtol=10e-2))
        self.assertTrue(
            torch.allclose(
                original_attn_weights, dp_attn_weights, atol=10e-4, rtol=10e-2
            )
        )

    def _run_multihead_x(self, **kwargs):
        x = torch.randn(16, 24, self.EMBED_SIZE)
        self._run_multihead(x, x, x, **kwargs)

    def _run_multihead_qkv(self, **kwargs):
        q = torch.randn(16, 24, self.EMBED_SIZE)
        k = torch.randn(20, 24, kwargs["kdim"] if "kdim" in kwargs else self.EMBED_SIZE)
        v = torch.randn(20, 24, kwargs["vdim"] if "vdim" in kwargs else self.EMBED_SIZE)
        self._run_multihead(q, k, v, **kwargs)

    def test_multihead_attention(self):
        for num_heads in (1, 2, 16):
            self._run_multihead_x(num_heads=num_heads)
            self._run_multihead_qkv(num_heads=num_heads)

            self._run_multihead_x(num_heads=num_heads, dropout=0.05)
            self._run_multihead_x(num_heads=num_heads, bias=False)
            self._run_multihead_x(num_heads=num_heads, add_bias_kv=True)
            self._run_multihead_x(num_heads=num_heads, bias=False, add_bias_kv=True)
            self._run_multihead_x(num_heads=num_heads, add_zero_attn=True)

            self._run_multihead_qkv(num_heads=num_heads, kdim=24, vdim=24)


class DPMultiheadAttentionChecker_test(unittest.TestCase):
    def setUp(self):
        self.checker = MultiheadAttentionChecker()
        self.EMBED_SIZE = 32
        self.DROPOUT = 0.5
        self.bias = True
        self.add_bias_kv = True
        self.kdim = 24
        self.vdim = 24
        self.num_heads = 4

        self.model = nn.MultiheadAttention(
            embed_dim=self.EMBED_SIZE,
            num_heads=self.num_heads,
            dropout=self.DROPOUT,
            add_bias_kv=self.add_bias_kv,
            kdim=self.kdim,
            vdim=self.vdim,
        )

    def test_raises_for_multihead_attn(self):
        with self.assertRaises(ShouldReplaceModuleError):
            self.checker.validate(self.model)

    def test_not_raises_for_no_multihead_attn(self):
        self.checker.validate(nn.Linear(2, 4))
        self.checker.validate(nn.Conv1d(4, 8, kernel_size=2))
        self.checker.validate(nn.Conv3d(12, 4, kernel_size=3))

    def test_replaces_module(self):
        replacement = self.checker.recommended_replacement(self.model)
        self.assertIsInstance(replacement, DPMultiheadAttention)

    def test_replacement_still_works(self):
        q = torch.randn(16, 24, self.EMBED_SIZE)
        k = torch.randn(20, 24, self.kdim)
        v = torch.randn(20, 24, self.vdim)

        y, attn_weights = self.model(q, k, v)
        replacement = self.checker.recommended_replacement(self.model)
        dp_y, dp_attn_weights = replacement(q, k, v)
        self.assertEqual(y.shape, dp_y.shape)
        self.assertEqual(attn_weights.shape, dp_attn_weights.shape)
