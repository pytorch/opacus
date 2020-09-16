#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test, ModelWithLoss


class InstanceNorm3d_test(GradSampleHooks_test):
    def test_5d_input(self):
        N, C, Z, H, W = 32, 3, 4, 8, 10
        x = torch.randn([N, C, Z, H, W])

        norm = nn.InstanceNorm3d(num_features=C, affine=True, track_running_stats=False)
        module = ModelWithLoss(norm, n_classes=C * Z * H * W)
        self.run_test(x, module, batch_first=True)
