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
        N=st.integers(0, 4),
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
        self.run_test(x, norm, batch_first=True, ew_compatible=N > 0)
