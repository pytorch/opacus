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
from ...utils.per_sample_gradients_utils import check_per_sample_gradients_are_correct, get_grad_sample_modes


class SequenceBias_test(GradSampleHooks_test):
    @given(
        N=st.integers(1, 4),
        T=st.integers(10, 20),
        D=st.integers(4, 8),
        batch_first=st.booleans(),
        test_or_check=st.integers(1, 2)
    )
    @settings(deadline=10000)
    def test_batch_second(
            self,
            N: int,
            T: int,
            D: int,
            batch_first: bool,
            test_or_check: int
    ):

        seqbias = SequenceBias(D, batch_first)
        if batch_first:
            x = torch.randn([N, T, D])
        else:
            x = torch.randn([T, N, D])
        if test_or_check == 1:
            self.run_test(x, seqbias, batch_first, ew_compatible=False)
        if test_or_check == 2:
            for grad_sample_mode in get_grad_sample_modes(use_ew=False):
                assert check_per_sample_gradients_are_correct(x, seqbias, batch_first=batch_first,
                                                              grad_sample_mode=grad_sample_mode)
