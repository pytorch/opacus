#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        self, N: int, Z: int, W: int, H: int, input_dim: int, norm_dim: int
    ):
        if norm_dim >= input_dim:
            return
        normalized_shape, x_shape = self.get_x_shape_and_norm_shape(
            H, N, W, Z, input_dim, norm_dim
        )

        norm = nn.LayerNorm(normalized_shape, elementwise_affine=True)
        x = torch.randn(x_shape)
        self.run_test(x, norm, batch_first=True)

    @staticmethod
    def get_x_shape_and_norm_shape(H, N, W, Z, input_dim, norm_dim):
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
        return normalized_shape, x_shape
