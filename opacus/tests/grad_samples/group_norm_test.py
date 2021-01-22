#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from .common import GradSampleHooks_test


class GroupNorm_test(GradSampleHooks_test):
    """
    We only test the case with ``affine=True`` here, because it is the only case that will actually
    compute a gradient. There is no grad_sample from this module otherwise.
    """

    def test_3d_input_one_group_affine(self):
        N, C, H, W = 32, 16, 8, 10
        x = torch.randn([N, C, H, W])
        norm = nn.GroupNorm(num_groups=1, num_channels=C, affine=True)
        self.run_test(x, norm, batch_first=True)

    def test_3d_input_four_groups_affine(self):
        N, C, H, W = 32, 16, 8, 10
        x = torch.randn([N, C, H, W])
        norm = nn.GroupNorm(num_groups=4, num_channels=C, affine=True)
        self.run_test(x, norm, batch_first=True)

    def test_3d_input_C_groups_affine(self):
        N, C, H, W = 32, 16, 8, 10
        x = torch.randn([N, C, H, W])
        norm = nn.GroupNorm(num_groups=C, num_channels=C, affine=True)
        self.run_test(x, norm, batch_first=True)
