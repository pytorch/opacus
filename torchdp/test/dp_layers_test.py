#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from torch.nn.modules.activation import MultiheadAttention
from torchdp.layers import DPMultiheadAttention


class DPLayersTest(unittest.TestCase):
    def setUp(self):
        self.EMBED_SIZE = 32

    def _reset_seeds(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)

    def _run_multihead(self, q, k, v, **kwargs):
        original_layer = MultiheadAttention(self.EMBED_SIZE, **kwargs)
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
