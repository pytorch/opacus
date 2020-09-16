#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test, ModelWithLoss


class InstanceNorm1d_test(GradSampleHooks_test):
    def test_3d_input(self):
        N, C, W = 32, 3, 10
        x = torch.randn([N, C, W])

        norm = nn.InstanceNorm1d(num_features=C, affine=True, track_running_stats=False)
        module = ModelWithLoss(norm, n_classes=C * W)
        self.run_test(x, module, batch_first=True)
