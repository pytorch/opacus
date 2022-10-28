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
from ...utils.per_sample_gradients_utils import (
    get_grad_sample_modes,
    check_per_sample_gradients_are_correct,
)


class Linear_test(GradSampleHooks_test):
    @given(
        N=st.integers(0, 4),
        Z=st.integers(1, 4),
        H=st.integers(1, 3),
        W=st.integers(10, 17),
        input_dim=st.integers(2, 4),
        bias=st.booleans(),
        batch_first=st.booleans(),
        test_or_check=st.integers(1, 2)
    )
    @settings(deadline=10000)
    def test_input_bias(
            self,
            N: int,
            Z: int,
            W: int,
            H: int,
            input_dim: int,
            bias: bool,
            batch_first: bool,
            test_or_check: int
    ):

        if input_dim == 2:
            if not batch_first:
                return  # see https://github.com/pytorch/opacus/pull/265
            else:
                x_shape = [N, W]
        if input_dim == 3:
            x_shape = [N, Z, W]
        if input_dim == 4:
            x_shape = [N, Z, H, W]

        linear = nn.Linear(W, W + 2, bias=bias)
        x = torch.randn(x_shape)
        if not batch_first:
            x = x.transpose(0, 1)
        ew_compatible = N > 0
        if test_or_check == 1:
            self.run_test(x, linear, batch_first=batch_first, ew_compatible=ew_compatible)
        if test_or_check == 2:
            for grad_sample_mode in get_grad_sample_modes(use_ew=ew_compatible):
                assert check_per_sample_gradients_are_correct(x, linear, batch_first=batch_first,
                                                              grad_sample_mode=grad_sample_mode)
