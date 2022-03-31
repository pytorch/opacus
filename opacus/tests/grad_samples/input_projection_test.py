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
from hypothesis import given, settings
from opacus.layers import InputProjection

from .common import GradSampleHooks_test


class InputProjection_test(GradSampleHooks_test):
    @given(
        embed_dim=st.integers(2, 4),
        kdim=st.integers(2, 4),
        vdim=st.integers(2, 4),
        bias=st.booleans(),
        batch_first=st.booleans(),
    )
    @settings(deadline=10000)
    def test_invariance_to_shape_and_batch_ordering(
        self, embed_dim: int, kdim: int, vdim: int, bias: bool, batch_first: bool
    ):
        input_projection = InputProjection(embed_dim, kdim, vdim, bias)
        query = torch.randn([1, 1, embed_dim])
        key = torch.randn([1, 1, kdim])
        value = torch.randn([1, 1, vdim])
        x = torch.cat((query, key, value), dim=-1)
        self.run_test(x, input_projection, batch_first=batch_first)
