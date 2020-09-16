#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test, ModelWithLoss


class Conv2d_test(GradSampleHooks_test):
    def test_4d_input(self):
        N, C, H, W = 32, 3, 8, 10
        x = torch.randn([N, C, H, W])

        conv_block = nn.Sequential(
            nn.Conv2d(C, 2 * C, kernel_size=2), nn.MaxPool2d(kernel_size=2)
        )

        with torch.no_grad():
            in_size = conv_block(x).size(-1)  # Maths too hard

        module = ModelWithLoss(conv_block, n_classes=in_size)
        self.run_test(x, module, batch_first=True)
