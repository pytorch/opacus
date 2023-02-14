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
from opacus.layers import SequenceBias

from .common import GradSampleHooks_test


class SequenceBias_test(GradSampleHooks_test):
    @given(
        N=st.integers(0, 4),
        T=st.integers(10, 20),
        D=st.integers(4, 8),
        batch_first=st.booleans(),
    )
    @settings(deadline=10000)
    def test_batch_second(self, N: int, T: int, D: int, batch_first: bool):
        seqbias = SequenceBias(D, batch_first)
        if batch_first:
            x = torch.randn([N, T, D])
        else:
            x = torch.randn([T, N, D])
        self.run_test(x, seqbias, batch_first, ew_compatible=False)
