#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test, ModelWithLoss


class InstanceNorm2d_test(GradSampleHooks_test):
    def test_4d_input(self):
        N, C, H, W = 32, 3, 8, 10
        x = torch.randn([N, C, H, W])

        norm = nn.InstanceNorm2d(num_features=C, affine=True, track_running_stats=False)
        module = ModelWithLoss(norm, n_classes=C * H * W)
        self.run_test(x, module, batch_first=True)
