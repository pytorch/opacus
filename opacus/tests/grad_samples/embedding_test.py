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


class Embedding_test(GradSampleHooks_test):
    @given(
        N=st.integers(0, 4),
        T=st.integers(1, 5),
        Q=st.integers(1, 4),
        R=st.integers(1, 2),
        V=st.integers(2, 32),
        D=st.integers(10, 17),
        dim=st.integers(2, 4),
        batch_first=st.booleans(),
    )
    @settings(deadline=10000)
    def test_input_across_dims(
        self,
        N: int,
        T: int,
        Q: int,
        R: int,
        V: int,
        D: int,
        dim: int,
        batch_first: bool,
    ):
        if dim == 1:  # TODO: fix when dim is 1
            size = [T]
        elif dim == 2:
            size = [N, T]
        elif dim == 3:
            size = [N, T, Q]
        elif dim == 4:
            size = [N, T, Q, R]

        emb = nn.Embedding(V, D)
        x = torch.randint(low=0, high=V, size=size)
        self.run_test(x, emb, batch_first=batch_first, ew_compatible=N > 0)
