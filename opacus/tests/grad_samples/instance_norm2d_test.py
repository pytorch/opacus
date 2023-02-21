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


class InstanceNorm2d_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        C=st.integers(1, 3),
        W=st.integers(5, 10),
        H=st.integers(4, 8),
    )
    @settings(deadline=10000)
    def test_4d_input(self, N: int, C: int, W: int, H: int):
        x = torch.randn([N, C, H, W])
        norm = nn.InstanceNorm2d(num_features=C, affine=True, track_running_stats=False)
        self.run_test(x, norm, batch_first=True)
