#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class InstanceNorm3d_test(GradSampleHooks_test):
    def test_5d_input(self):
        N, C, Z, H, W = 32, 3, 4, 8, 10
        x = torch.randn([N, C, Z, H, W])
        norm = nn.InstanceNorm3d(num_features=C, affine=True, track_running_stats=False)
        self.run_test(x, norm, batch_first=True)
