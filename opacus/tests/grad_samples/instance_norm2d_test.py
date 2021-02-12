#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from typing import Optional, Tuple, Callable, Union

import hypothesis.strategies as st
from hypothesis import given, settings

from .common import GradSampleHooks_test


class InstanceNorm2d_test(GradSampleHooks_test):
    @given(
        N=st.integers(20, 32),
        C=st.integers(1,3),
        W=st.integers(5,10),
        H=st.integers(4,8),
    )
    @settings(deadline=10000)
    def test_4d_input(
        self,
        N: int,
        C: int,
        W: int,
        H: int,
    ):

        x = torch.randn([N, C, H, W])
        norm = nn.InstanceNorm2d(num_features=C, affine=True, track_running_stats=False)
        self.run_test(x, norm, batch_first=True)
