#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class Linear_test(GradSampleHooks_test):
    def test_2d_input_bias(self):
        N, W = 32, 17
        linear = nn.Linear(W, W + 2, bias=True)
        x = torch.randn([N, W])
        self.run_test(x, linear, batch_first=True)

    def test_3d_input_bias(self):
        N, Z, W = 32, 4, 10
        linear = nn.Linear(W, W + 2, bias=True)
        x = torch.randn([N, Z, W])
        self.run_test(x, linear, batch_first=True)

    def test_4d_input_bias(self):
        N, Z, Q, W = 32, 4, 3, 10
        linear = nn.Linear(W, W + 2, bias=True)
        x = torch.randn([N, Z, Q, W])
        self.run_test(x, linear, batch_first=True)

    def test_2d_input_nobias(self):
        N, W = 32, 17
        linear = nn.Linear(W, W + 2, bias=False)
        x = torch.randn([N, W])
        self.run_test(x, linear, batch_first=True)

    def test_3d_input_nobias(self):
        N, Z, W = 32, 4, 10
        linear = nn.Linear(W, W + 2, bias=False)
        x = torch.randn([N, Z, W])
        self.run_test(x, linear, batch_first=True)

    def test_4d_input_nobias(self):
        N, Z, Q, W = 32, 4, 3, 10
        linear = nn.Linear(W, W + 2, bias=False)
        x = torch.randn([N, Z, Q, W])
        self.run_test(x, linear, batch_first=True)
