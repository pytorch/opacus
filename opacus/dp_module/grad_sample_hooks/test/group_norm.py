#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test, ModelWithLoss


class GroupNorm_test(GradSampleHooks_test):
    def test_3d_input_one_group(self):
        N, C, H, W = 32, 16, 8, 10
        x = torch.randn([N, C, H, W])

        norm = nn.GroupNorm(num_groups=1, num_channels=C, affine=True)
        module = ModelWithLoss(norm, n_classes=C * H * W)
        self.run_test(x, module, batch_first=True)

    def test_3d_input_four_groups(self):
        N, C, H, W = 32, 16, 8, 10
        x = torch.randn([N, C, H, W])

        norm = nn.GroupNorm(num_groups=4, num_channels=C, affine=True)
        module = ModelWithLoss(norm, n_classes=C * H * W)
        self.run_test(x, module, batch_first=True)

    def test_3d_input_C_groups(self):
        N, C, H, W = 32, 16, 8, 10
        x = torch.randn([N, C, H, W])

        norm = nn.GroupNorm(num_groups=C, num_channels=C, affine=True)
        module = ModelWithLoss(norm, n_classes=C * H * W)
        self.run_test(x, module, batch_first=True)
