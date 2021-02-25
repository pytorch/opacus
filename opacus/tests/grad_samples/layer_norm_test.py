#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test


class LayerNorm_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        Z=st.integers(1, 4),
        H=st.integers(1, 3),
        W=st.integers(5, 10),
        input_dim=st.integers(2, 4),
        norm_dim=st.integers(1, 3),
    )
    @settings(deadline=10000)
    def test_input_norm(
        self,
        N: int,
        Z: int,
        W: int,
        H: int,
        input_dim: int,
        norm_dim: int,
    ):

        if norm_dim >= input_dim:
            return
        if norm_dim == 1:
            normalized_shape = W
            if input_dim == 2:
                x_shape = [N, W]
            if input_dim == 3:
                x_shape = [N, Z, W]
            if input_dim == 4:
                x_shape = [N, Z, H, W]
        elif norm_dim == 2:
            if input_dim == 3:
                normalized_shape = [Z, W]
                x_shape = [N, Z, W]
            if input_dim == 4:
                normalized_shape = [H, W]
                x_shape = [N, Z, H, W]
        elif norm_dim == 3:
            normalized_shape = [Z, H, W]
            x_shape = [N, Z, H, W]

        norm = nn.LayerNorm(normalized_shape, elementwise_affine=True)
        x = torch.randn(x_shape)
        self.run_test(x, norm, batch_first=True)
