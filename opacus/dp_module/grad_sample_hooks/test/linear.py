#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test, ModelWithLoss


class Linear_test(GradSampleHooks_test):
    def test_2d_input(self):
        N, W = 32, 17
        linear = nn.Linear(W, W + 2)
        module = ModelWithLoss(linear, n_classes=W + 2)
        x = torch.randn([N, W])
        self.run_test(x, module, batch_first=True)

    def test_3d_input(self):
        N, Z, W = 32, 4, 10
        linear = nn.Linear(W, W + 2)
        module = ModelWithLoss(linear, n_classes=Z * (W + 2))
        x = torch.randn([N, Z, W])
        self.run_test(x, module, batch_first=True)

    def test_4d_input(self):
        N, Z, Q, W = 32, 4, 3, 10
        linear = nn.Linear(W, W + 2)
        module = ModelWithLoss(linear, n_classes=Z * Q * (W + 2))
        x = torch.randn([N, Z, Q, W])
        self.run_test(x, module, batch_first=True)
