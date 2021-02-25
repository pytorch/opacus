#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test


class GroupNorm_test(GradSampleHooks_test):
    """
    We only test the case with ``affine=True`` here, because it is the only case that will actually
    compute a gradient. There is no grad_sample from this module otherwise.
    """

    @given(
        N=st.integers(1, 4),
        C=st.integers(1, 8),
        H=st.integers(5, 10),
        W=st.integers(4, 8),
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

        if C % num_groups != 0:
            return

        x = torch.randn([N, C, H, W])
        norm = nn.GroupNorm(num_groups=num_groups, num_channels=C, affine=True)
        self.run_test(x, norm, batch_first=True)
