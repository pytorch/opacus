#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from typing import Optional, Tuple, Callable, Union

import hypothesis.strategies as st
from hypothesis import given, settings

from .common import GradSampleHooks_test


class GroupNorm_test(GradSampleHooks_test):
    """
    We only test the case with ``affine=True`` here, because it is the only case that will actually
    compute a gradient. There is no grad_sample from this module otherwise.
    """
    @given(
        N=st.sampled_from([32]),
        C=st.sampled_from([16]),
        H=st.sampled_from([8]),
        W=st.sampled_from([10]),
        num_groups=st.sampled_from([1, 4, "C"]),
    )
    @settings(deadline=10000)
    def test_3d_input_groups(
        self,
        N: int,
        C: int,
        H: int,
        W: int,
        num_groups: Union[int, str],
    ):

        if num_groups == "C":
            num_groups = C

        x = torch.randn([N, C, H, W])
        norm = nn.GroupNorm(num_groups=num_groups, num_channels=C, affine=True)
        self.run_test(x, norm, batch_first=True)
