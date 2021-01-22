#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class InstanceNorm1d_test(GradSampleHooks_test):
    def test_3d_input(self):
        N, C, W = 32, 3, 10
        x = torch.randn([N, C, W])
        norm = nn.InstanceNorm1d(num_features=C, affine=True, track_running_stats=False)
        self.run_test(x, norm, batch_first=True)
