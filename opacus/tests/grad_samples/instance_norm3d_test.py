#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from typing import Optional, Tuple, Callable, Union

import hypothesis.strategies as st
from hypothesis import given, settings

from .common import GradSampleHooks_test


class InstanceNorm3d_test(GradSampleHooks_test):
    @given(
        N=st.sampled_from([32]),
        C=st.sampled_from([3]),
        W=st.sampled_from([10]),
        H=st.sampled_from([8]),
        Z=st.sampled_from([4]),
    )
    @settings(deadline=10000)
    def test_5d_input(
        self,
        N: int,
        C: int,
        W: int,
        H: int,
        Z: int,
    ):
        x = torch.randn([N, C, Z, H, W])
        norm = nn.InstanceNorm3d(num_features=C, affine=True, track_running_stats=False)
        self.run_test(x, norm, batch_first=True)
