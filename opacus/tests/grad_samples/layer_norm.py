#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class LayerNorm_test(GradSampleHooks_test):
    def test_2d_input_1d_norm(self):
        N, W = 32, 17
        norm = nn.LayerNorm(W, elementwise_affine=True)
        x = torch.randn([N, W])
        self.run_test(x, norm, batch_first=True)

    def test_3d_input_1d_norm(self):
        N, Z, W = 32, 4, 10
        norm = nn.LayerNorm(W, elementwise_affine=True)
        x = torch.randn([N, Z, W])
        self.run_test(x, norm, batch_first=True)

    def test_3d_input_2d_norm(self):
        N, Z, W = 32, 4, 10
        norm = nn.LayerNorm([Z, W], elementwise_affine=True)
        x = torch.randn([N, Z, W])
        self.run_test(x, norm, batch_first=True)

    def test_4d_input_1d_norm(self):
        N, C, H, W = 32, 4, 3, 10
        norm = nn.LayerNorm(W, elementwise_affine=True)
        x = torch.randn([N, C, H, W])
        self.run_test(x, norm, batch_first=True)

    def test_4d_input_2d_norm(self):
        N, C, H, W = 32, 4, 3, 10
        norm = nn.LayerNorm([H, W], elementwise_affine=True)
        x = torch.randn([N, C, H, W])
        self.run_test(x, norm, batch_first=True)

    def test_4d_input_3d_norm(self):
        N, C, H, W = 32, 4, 3, 10
        norm = nn.LayerNorm([C, H, W], elementwise_affine=True)
        x = torch.randn([N, C, H, W])
        self.run_test(x, norm, batch_first=True)
