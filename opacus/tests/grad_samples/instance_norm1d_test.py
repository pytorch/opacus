#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test


class InstanceNorm1d_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        C=st.integers(1, 3),
        W=st.integers(5, 10),
    )
    @settings(deadline=10000)
    def test_3d_input(
        self,
        N: int,
        C: int,
        W: int,
    ):

        x = torch.randn([N, C, W])
        norm = nn.InstanceNorm1d(num_features=C, affine=True, track_running_stats=False)
        self.run_test(x, norm, batch_first=True)
